import argparse
import yaml
import subprocess
import os
import sys
import shutil
import copy
from dllm.utils.naming import get_experiment_naming, flatten_config_dict
from dllm.utils.config import load_resolved_config, resolve_keywords, expand_matrix_config

def resolve_config_path(path: str, base_dir: str) -> str:
    """
    Resolves a config name to a full path.
    1. If path exists as-is, return it.
    2. If base_dir/path exists, return it.
    3. If base_dir/path.yaml exists, return it.
    """
    if os.path.exists(path):
        return path
    
    # Try directory-specific resolution
    p = os.path.join(base_dir, path)
    if os.path.exists(p):
        return p
    if not p.endswith(".yaml") and os.path.exists(p + ".yaml"):
        return p + ".yaml"
    
    return path # fall back to original for error reporting

def main():
    parser = argparse.ArgumentParser(description="Central Training Launcher for dLLM")
    
    # Core Config Paths
    parser.add_argument("run_config", help="Name or path of training/wandb configuration YAML")
    parser.add_argument("--slurm_config", 
                        default="default",
                        help="Name or path of Slurm resource configuration YAML")
    parser.add_argument("--accelerate_config", 
                        default="fsdp", 
                        help="Name of accelerate config (located in configs/accelerate/)")
    parser.add_argument("--job_name", 
                        default="dllm", 
                        help="Slurm job name")
    parser.add_argument("--begin",
                        help="Time to start the Slurm job (e.g. 'now+60' or '22:00')")
    parser.add_argument("--dry_run", action="store_true", help="Generate scripts but do not submit jobs")
    
    # Collect remaining args to pass directly to training script
    args, extra_args = parser.parse_known_args()

    # 0. Load Keywords
    keyword_map = {}
    if os.path.exists("configs/keywords.yaml"):
        with open("configs/keywords.yaml", "r") as f:
            keyword_map = yaml.safe_load(f) or {}

    # 1. Load Slurm Config (Shared across all matrix entries)
    slurm_path = resolve_config_path(args.slurm_config, "configs/slurm")
    with open(slurm_path, 'r') as f:
        slurm_cfg_base = yaml.safe_load(f) or {}
    slurm_cfg_base["job_name"] = args.job_name
    if args.begin:
        slurm_cfg_base["begin"] = args.begin

    # 2. Load and Resolve Training Config(s)
    # This automatically merges with configs/default.yaml if it exists
    base_run_cfg = load_resolved_config(args.run_config, "configs/train", "../default.yaml")
    if "training" in base_run_cfg:
        base_run_cfg["training"] = flatten_config_dict(base_run_cfg["training"])
    
    # 3. Handle CLI Overwrites (Apply to base before expansion)
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--slurm.") or arg.startswith("--training."):
            if "=" in arg:
                key_full, val = arg.split("=", 1)
            else:
                key_full = arg
                val = extra_args[i+1] if i+1 < len(extra_args) else True
                i += 1
            
            prefix, key = key_full.lstrip("-").split(".", 1)
            
            # Type conversion for common types
            if isinstance(val, str):
                if val.lower() == "true": val = True
                elif val.lower() == "false": val = False
                elif val.startswith("[") and val.endswith("]"):
                    import ast
                    try: val = ast.literal_eval(val)
                    except: pass

            if prefix == "slurm":
                slurm_cfg_base[key] = val
            elif prefix == "training":
                if "training" not in base_run_cfg: base_run_cfg["training"] = {}
                base_run_cfg["training"][key] = val
        i += 1

    # Resolve keywords in the base config after all CLI overrides are applied
    base_run_cfg = resolve_keywords(base_run_cfg, keyword_map)

    # 4. Expand Matrix of Experiments
    try:
        exp_configs = list(expand_matrix_config(base_run_cfg))
    except ValueError as e:
        print(f"❌ Configuration Matrix Error: {e}")
        sys.exit(1)
        
    print(f"🧪 Found {len(exp_configs)} experiment(s) in the matrix configuration.")

    for idx, run_cfg in enumerate(exp_configs):
        slurm_cfg = copy.deepcopy(slurm_cfg_base)
        
        # 4a. Naming and Directories
        if "training" not in run_cfg: run_cfg["training"] = {}
        if "seed" not in run_cfg["training"]: run_cfg["training"]["seed"] = 42
        
        group, run_name, tags, output_dir = get_experiment_naming(run_cfg, slurm_cfg)
        
        # If matrix, append index to job name to distinguish in squeue
        if len(exp_configs) > 1:
            slurm_cfg["job_name"] = f"{slurm_cfg['job_name']}_{idx}"

        # 4b. Construct Slurm Directives
        slurm_directives = ["#!/bin/bash"]
        sbatch_map = {
            "begin": "--begin",
            "job_name": "--job-name",
            "nodes": "--nodes",
            "gpus_per_node": "--gpus-per-node",
            "time": "--time",
            "mem": "--mem",
            "cpus_per_task": "--cpus-per-task",
            "ntasks_per_node": "--ntasks-per-node",
            "partition": "--partition",
            "output": "--output",
            "error": "--error",
            "requeue": "--requeue",
            "working_dir": "--chdir",
            "account": "--account",
            "signal": "--signal"
        }
        for k, v in slurm_cfg.items():
            if k in sbatch_map:
                flag = sbatch_map[k]
                if v is True: slurm_directives.append(f"#SBATCH {flag}")
                elif v is False: continue
                else:
                    if flag.startswith("--"): slurm_directives.append(f"#SBATCH {flag}={v}")
                    else: slurm_directives.append(f"#SBATCH {flag} {v}")

        # 4c. Setup Environment Variables
        env_exports = [
            f"export WANDB_NAME=\"{run_name}\"",
            f"export WANDB_RUN_GROUP=\"{group}\"",
            f"export WANDB_TAGS=\"{','.join(tags)}\"",
            f"export WANDB_PROJECT=\"{os.getenv('WANDB_PROJECT', 'BPTT-llada')}\"",
            "export WANDB_INIT_TIMEOUT=300",
            "export HF_HOME=\"$PWD/.cache\"",
            "export HF_DATASETS_CACHE=\"$PWD/.cache/datasets\"",
            "export PYTHONHASHSEED=42",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=43200", 
            "export ACCELERATE_TIMEOUT_IN_SECONDS=43200", 
        ]

        # 4d. Prepare training flags
        training = run_cfg["training"]
        script_path = training.pop("script_path", "examples/llada/sft.py")
        training["output_dir"] = output_dir
        
        train_flags = []
        for k, v in training.items():
            if v is True: train_flags.append(f"--{k}")
            elif v is False: train_flags.append(f"--{k} False")
            elif v is None: continue
            else:
                if isinstance(v, list):
                    v = " ".join(map(str, v))
                train_flags.append(f"--{k} {v}")
        
        # Add remaining pass-through args
        for arg in extra_args:
            if not arg.startswith("--slurm.") and not arg.startswith("--training."):
                train_flags.append(arg)

        # 4e. Modeling setup
        if "model_name_or_path" in training:
            model_src = training["model_name_or_path"]
            if os.path.exists(model_src) and os.path.isdir(model_src):
                if not args.dry_run:
                    print(f"ℹ️ Copying LLaDA modeling files to {output_dir}...")
                    os.makedirs(output_dir, exist_ok=True)
                    for f in os.listdir(model_src):
                        if f.endswith(".py"):
                            shutil.copy2(os.path.join(model_src, f), output_dir)

        # 4f. Generate Script
        acc_config = args.accelerate_config
        if not acc_config.endswith(".yaml"):
            acc_config = f"configs/accelerate/{acc_config}.yaml"
        working_dir = slurm_cfg.get("working_dir", os.getcwd())
        
        slurm_header = "\n".join(slurm_directives)
        train_args_str = " ".join(train_flags)
        env_exports_str = "\n".join(env_exports)
        
        # Fix f-string backslash issue by pre-calculating or avoiding backslashes in {}
        bash_script = f"""{slurm_header}
set -e
cd {working_dir}
module load StdEnv/2023 python/3.11.5 cuda/12.6 cudnn gcc opencv arrow
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -a
[ -f .env ] && . ./.env
set +a
{env_exports_str}

MASTER_NAME=$(scontrol show hostnames "${{SLURM_JOB_NODELIST:-localhost}}" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NAME hostname -i | awk '{{for(i=1;i<=NF;i++) if($i ~ /^10\\.[0-9]+\\./) {{print $i; exit}}}}')
MASTER_ADDR=${{MASTER_ADDR:-$MASTER_NAME}}
MASTER_PORT=$((20000 + ${{SLURM_JOB_ID:-0}} % 10000))

WORLD_SIZE=$((${{SLURM_NNODES:-1}} * ${{SLURM_GPUS_ON_NODE:-1}}))
LAUNCH_ARGS="--num_processes $WORLD_SIZE"
if [ "$WORLD_SIZE" -gt 1 ]; then
    LAUNCH_ARGS="--multi_gpu $LAUNCH_ARGS --num_machines ${{SLURM_NNODES:-1}}"
    if [ "${{SLURM_NNODES:-1}}" -gt 1 ]; then
        LAUNCH_ARGS="$LAUNCH_ARGS --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --machine_rank \$SLURM_NODEID --rdzv_backend c10d"
    fi
    if [[ "{acc_config}" == *"fsdp"* ]]; then LAUNCH_ARGS="$LAUNCH_ARGS --fsdp_transformer_layer_cls_to_wrap LLaDABlock,Fast_dLLM_QwenDecoderLayer"; fi
fi

srun --label --ntasks-per-node=1 bash -c "accelerate launch $LAUNCH_ARGS --config_file \"{acc_config}\" \"{script_path}\" {train_args_str}"
"""
        script_name = f".generated_train_{idx}.sh"
        if len(exp_configs) == 1: script_name = ".generated_train.sh"

        with open(script_name, "w") as f:
            f.write(bash_script)
        
        # 5. Submission
        if args.dry_run:
            print(f"🔍 [Dry Run] Generated {script_name} for experiment: {run_name}")
            continue
            
        print(f"🚀 Submitting experiment {idx+1}/{len(exp_configs)}: {run_name}")
        os.makedirs(".logs", exist_ok=True)
        result = subprocess.run(["sbatch", script_name])
        if result.returncode == 0:
            print(f"✅ Job submitted. Output directory: {output_dir}")
        else:
            print(f"❌ Failed to submit job {idx}.")

if __name__ == "__main__":
    main()
