import argparse
import yaml
import subprocess
import os
import sys
import copy
from dllm.utils.naming import get_eval_naming, flatten_config_dict
from dllm.utils.config import load_resolved_config, resolve_keywords, expand_matrix_config

def dict_to_arg_str(d, sep=","):
    """Convert dict to 'key=val,key2=val2' format."""
    return sep.join([f"{k}={v}" for k, v in d.items()])

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
    parser = argparse.ArgumentParser(description="Central Evaluation Launcher for dLLM")
    
    # Core Config Paths
    parser.add_argument("run_config", help="Name or path of evaluation configuration")
    parser.add_argument("--slurm_config", required=True, help="Name or path of Slurm resource configuration")
    parser.add_argument("--accelerate_config", default="ddp", help="Name of accelerate config (located in configs/accelerate/)")
    parser.add_argument("--job_name", default="dllm_eval", help="Slurm job name")
    parser.add_argument("--begin", help="Time to start the job (Slurm format, e.g. now+2hours)")
    parser.add_argument("--after", help="Slurm job ID to wait for (uses --dependency=afterok:<id>)")
    parser.add_argument("--dry_run", action="store_true", help="Generate scripts but do not submit jobs")
    
    # Collect remaining args to pass directly to the eval script
    args, extra_args = parser.parse_known_args()

    # 0. Load Keywords
    keyword_map = {}
    if os.path.exists("configs/keywords.yaml"):
        with open("configs/keywords.yaml", "r") as f:
            keyword_map = yaml.safe_load(f) or {}

    # 1. Load Slurm Config (Shared)
    slurm_path = resolve_config_path(args.slurm_config, "configs/slurm")
    with open(slurm_path, 'r') as f:
        slurm_cfg_base = yaml.safe_load(f) or {}
    slurm_cfg_base["job_name"] = args.job_name

    # 2. Load and Resolve Evaluation Config(s)
    base_run_cfg = load_resolved_config(args.run_config, "configs/eval", "../default.yaml")
    if "evaluation" in base_run_cfg:
        base_run_cfg["evaluation"] = flatten_config_dict(base_run_cfg["evaluation"])
    
    # Resolve keywords (e.g. math -> full dataset string)
    base_run_cfg = resolve_keywords(base_run_cfg, keyword_map)

    # 3. CLI Overwrites
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--slurm.") or arg.startswith("--evaluation."):
            if "=" in arg:
                key_full, val = arg.split("=", 1)
            else:
                key_full = arg
                val = extra_args[i+1] if i+1 < len(extra_args) else True
                i += 1
            prefix, key = key_full.lstrip("-").split(".", 1)
            if prefix == "slurm":
                slurm_cfg_base[key] = val
            elif prefix == "evaluation":
                if "evaluation" not in base_run_cfg: base_run_cfg["evaluation"] = {}
                base_run_cfg["evaluation"][key] = val
        i += 1

    # 4. Expand Matrix
    try:
        exp_configs = list(expand_matrix_config(base_run_cfg))
    except ValueError as e:
        print(f"❌ Configuration Matrix Error: {e}")
        sys.exit(1)
        
    print(f"🧪 Found {len(exp_configs)} evaluation(s) in the matrix configuration.")

    for idx, run_cfg in enumerate(exp_configs):
        slurm_cfg = copy.deepcopy(slurm_cfg_base)
        evaluation = run_cfg.get("evaluation", {})
        
        # Distinguish job name if matrix
        if len(exp_configs) > 1:
            slurm_cfg["job_name"] = f"{slurm_cfg['job_name']}_{idx}"

        # 4b. Slurm Directives
        slurm_directives = ["#!/bin/bash"]
        sbatch_map = {
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
            "account": "--account"
        }
        for k, v in slurm_cfg.items():
            if k in sbatch_map:
                flag = sbatch_map[k]
                if v is True: slurm_directives.append(f"#SBATCH {flag}")
                elif v is False: continue
                else:
                    if flag.startswith("--"): slurm_directives.append(f"#SBATCH {flag}={v}")
                    else: slurm_directives.append(f"#SBATCH {flag} {v}")
        if args.after:
            slurm_directives.append(f"#SBATCH --dependency=afterok:{args.after}")
        if args.begin:
             slurm_directives.append(f"#SBATCH --begin={args.begin}")

        # 4c. Setup Environment Variables
        if "seed" not in evaluation: evaluation["seed"] = 42
        seed = evaluation["seed"]
        
        wb = evaluation.get("wandb_args", {})
        env_exports = [
            f"export WANDB_NAME=\"{wb.get('name', 'eval')}\"",
            f"export WANDB_RUN_GROUP=\"{wb.get('group', 'evals')}\"",
            f"export WANDB_PROJECT=\"{os.getenv('WANDB_PROJECT', 'BPTT-llada')}\"",
            f"export PYTHONHASHSEED={seed}",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "export HF_HOME=\"$PWD/.cache\"",
            "export HF_DATASETS_CACHE=\"$PWD/.cache/datasets\"",
            "export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600",
            "export ACCELERATE_TIMEOUT_IN_SECONDS=3600",
        ]

        # 4d. Prepare eval flags
        script_path = evaluation.pop("script_path", "dllm/pipelines/llada/eval.py")
        model_args = evaluation.get("model_args", {})
        
        # Handle manual model_args overrides from command line if any
        for arg in extra_args:
             if "--model_args" in arg:
                arg_content = arg.split("--model_args")[-1].strip().strip("\"'")
                for kv in arg_content.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        model_args[k.strip()] = v.strip()
        evaluation["model_args"] = model_args
        
        task_slug, model_slug, checkpoint_name, params_slug, output_path = get_eval_naming(evaluation)
        
        eval_flags = []
        for k, v in evaluation.items():
            if k == "model_args":
                eval_flags.append(f"--model_args \"{dict_to_arg_str(v)}\"")
            elif v is True: eval_flags.append(f"--{k}")
            elif v is False or v is None: continue
            else: eval_flags.append(f"--{k} {v}")
        
        # Finish exports with run-specific naming info
        env_exports.append(f"export WANDB_TAGS=\"{model_slug}|{task_slug}|{params_slug}\"")

        # 4f. Generate Script
        acc_config = args.accelerate_config
        if not acc_config.endswith(".yaml"):
            acc_config = f"configs/accelerate/{acc_config}.yaml"
        working_dir = slurm_cfg.get("working_dir", os.getcwd())
        
        slurm_header = "\n".join(slurm_directives)
        eval_args_str = " ".join(eval_flags)
        env_exports_str = "\n".join(env_exports)
        
        bash_script = f"""{slurm_header}
set -e
cd {working_dir}
module load StdEnv/2023 python/3.11.5 cuda/12.6 cudnn gcc opencv arrow
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -a
[ -f .env ] && . ./.env
set +a
{env_exports_str}

WORLD_SIZE=$((${{SLURM_NNODES:-1}} * ${{SLURM_GPUS_ON_NODE:-1}}))
LAUNCH_ARGS="--num_processes $WORLD_SIZE"
if [ "$WORLD_SIZE" -gt 1 ]; then
    LAUNCH_ARGS="--multi_gpu $LAUNCH_ARGS --num_machines ${{SLURM_NNODES:-1}}"
fi

srun --label --ntasks-per-node=1 bash -c "accelerate launch $LAUNCH_ARGS --config_file \"{acc_config}\" \"{script_path}\" {eval_args_str}"
"""
        script_name = f".generated_eval_{idx}.sh"
        if len(exp_configs) == 1: script_name = ".generated_eval.sh"
        
        with open(script_name, "w") as f:
            f.write(bash_script)
            
        if args.dry_run:
            print(f"🔍 [Dry Run] Generated {script_name} for evaluation: {task_slug} on {model_slug}")
            continue
            
        print(f"🚀 Submitting evaluation {idx+1}/{len(exp_configs)}: {task_slug}")
        os.makedirs(".logs", exist_ok=True)
        result = subprocess.run(["sbatch", script_name])
        if result.returncode == 0:
            print(f"✅ Job submitted. Results will be at: {output_path}")
        else:
            print(f"❌ Failed to submit job {idx}.")

if __name__ == "__main__":
    main()
