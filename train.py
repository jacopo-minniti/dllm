import argparse
import yaml
import subprocess
import os
import sys
from dllm.utils.naming import get_experiment_naming

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
    
    # Collect remaining args to pass directly to training script
    args, extra_args = parser.parse_known_args()

    # Resolve paths
    args.run_config = resolve_config_path(args.run_config, "configs/train")
    args.slurm_config = resolve_config_path(args.slurm_config, "configs/slurm")

    # 1. Load YAML Configurations
    with open(args.run_config, 'r') as f:
        run_cfg = yaml.safe_load(f)
    with open(args.slurm_config, 'r') as f:
        slurm_cfg = yaml.safe_load(f)

    # 1a. Set default job name or override from CLI
    slurm_cfg["job_name"] = args.job_name

    # 1b. Handle CLI Overwrites (Format: --slurm.key val or --training.key val)
    remaining_extra_args = []
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--slurm.") or arg.startswith("--training."):
            # handle --key=val case
            if "=" in arg:
                key_full, val = arg.split("=", 1)
            else:
                key_full = arg
                val = extra_args[i+1] if i+1 < len(extra_args) else True
                i += 1
            
            prefix, key = key_full.lstrip("-").split(".", 1)
            
            # Apply override
            if prefix == "slurm":
                slurm_cfg[key] = val
                print(f"🔧 Overwriting Slurm config: {key} = {val}")
            elif prefix == "training":
                if "training" not in run_cfg: run_cfg["training"] = {}
                run_cfg["training"][key] = val
                print(f"🔧 Overwriting Training config: {key} = {val}")
        else:
            remaining_extra_args.append(arg)
        i += 1
    extra_args = remaining_extra_args

    # 2. Extract Slurm Directives and Setup Metadata
    # 2a. Determine naming based on config
    if "training" not in run_cfg: run_cfg["training"] = {}
    if "seed" not in run_cfg["training"]: run_cfg["training"]["seed"] = 42
    
    group, name, tags, output_dir, run_id = get_experiment_naming(run_cfg, slurm_cfg)
    
    # 2b. Slurm mapping
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
    }
    for k, v in slurm_cfg.items():
        if k in sbatch_map:
            flag = sbatch_map[k]
            if v is True:
                slurm_directives.append(f"#SBATCH {flag}")
            elif v is False:
                continue
            else:
                if flag.startswith("--"):
                    slurm_directives.append(f"#SBATCH {flag}={v}")
                else:
                    slurm_directives.append(f"#SBATCH {flag} {v}")

    # 4. Prepare Training Command
    training = run_cfg.get("training", {})
    script_path = training.pop("script_path", "examples/llada/sft.py")
    
    # 3. Setup WandB and Environment Variables
    env_exports = [
        f"export WANDB_NAME=\"{name}\"",
        f"export WANDB_RUN_GROUP=\"{group}\"",
        f"export WANDB_RUN_ID=\"{run_id}\"",
        f"export WANDB_RESUME=\"allow\"",
        f"export WANDB_TAGS=\"{','.join(tags)}\"",
        f"export WANDB_PROJECT=\"{os.getenv('WANDB_PROJECT', 'BPTT-llada')}\"",
        "export WANDB_INIT_TIMEOUT=300",
        "export WANDB_DEBUG=false",
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
    ]

    # 4. Prepare Training Command
    training = run_cfg["training"]
    script_path = training.pop("script_path", "examples/llada/sft.py")
    training["output_dir"] = output_dir
    
    # 4a. Auto-resume logic
    if os.path.exists(output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(output_dir)):
        if "resume_from_checkpoint" not in training:
            training["resume_from_checkpoint"] = "True"
            print(f"🔄 Checkpoint found in {output_dir}. Auto-resuming...")
    
    # Combine training params from YAML and CLI extra args
    train_flags = []
    for k, v in training.items():
        if isinstance(v, bool):
            if v: train_flags.append(f"--{k}")
        else:
            train_flags.append(f"--{k}")
            train_flags.append(str(v))
    train_flags.extend(extra_args)

    # 4b. Automation: Ensure custom modeling code is present in the output directory
    # This ensures and enables `trust_remote_code=True` loading for checkpoints.
    if "llada" in training.get("model_name_or_path", "").lower():
        model_src = "dllm/pipelines/llada/models/"
        if os.path.exists(model_src):
            print(f"ℹ️ Copying LLaDA modeling files to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
            for f in os.listdir(model_src):
                if f.endswith(".py"):
                    shutil.copy2(os.path.join(model_src, f), output_dir)

    # Handle accelerate config path
    acc_config = args.accelerate_config
    if not acc_config.endswith(".yaml"):
        acc_config = f"configs/accelerate/{acc_config}.yaml"

    # Get working directory for script activation
    working_dir = slurm_cfg.get("working_dir", os.getcwd())

    # 5. Generate the Slurm Bash Script
    bash_script = f"""{chr(10).join(slurm_directives)}
set -e

# ===== System Environment =====
echo "Running from: $(pwd)"
cd {working_dir}
echo "Switched to: $(pwd)"

module load StdEnv/2023 python/3.11.5 cuda/12.6 cudnn
module load gcc opencv arrow

# Activate virtualenv
if [ -f "{working_dir}/.venv/bin/activate" ]; then
    source "{working_dir}/.venv/bin/activate"
elif [ -f "./.venv/bin/activate" ]; then
    source "./.venv/bin/activate"
else
    echo "Warning: Could not find .venv/bin/activate"
fi

set -a
[ -f .env ] && . ./.env
set +a

# ===== Exports =====
{chr(10).join(env_exports)}
export NCCL_SOCKET_IFNAME=eth0,enp,eno
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# ===== Scale Calculation =====
NUM_NODES=${{SLURM_NNODES:-1}}
GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\\n' | wc -l)
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames "${{SLURM_JOB_NODELIST:-localhost}}" | head -n 1)
MASTER_PORT=$((20000 + ${{SLURM_JOB_ID:-0}} % 10000))

if [ -z "$SLURM_JOB_ID" ]; then
    MASTER_ADDR="localhost"
    MASTER_PORT=29500
    SLURM_PROCID=0
fi

echo "Launching: NUM_NODES=$NUM_NODES, GPUS=$WORLD_SIZE on $MASTER_ADDR:$MASTER_PORT (via srun)"

# ===== Execution =====
srun --ntasks-per-node=1 --nodes="${{NUM_NODES}}" bash -c "accelerate launch \\
  --config_file \"{acc_config}\" \\
  --num_machines \"${{NUM_NODES}}\" \\
  --num_processes \"${{WORLD_SIZE}}\" \\
  --main_process_ip \"${{MASTER_ADDR}}\" \\
  --main_process_port \"${{MASTER_PORT}}\" \\
  --machine_rank \"\$SLURM_NODEID\" \\
  --rdzv_backend c10d \\
  \"{script_path}\" {" ".join(train_flags)}"
"""

    # Write to a temporary file
    temp_script = ".generated_train.sh"
    with open(temp_script, "w") as f:
        f.write(bash_script)
    
    # Submit job
    os.makedirs(".logs", exist_ok=True)
    print(f"🚀 Submitting job via configuration: {args.run_config}")
    result = subprocess.run(["sbatch", temp_script])
    
    if result.returncode == 0:
        print(f"✅ Job submitted successfully. Script: {temp_script}")
    else:
        print(f"❌ Failed to submit job.")

if __name__ == "__main__":
    main()
