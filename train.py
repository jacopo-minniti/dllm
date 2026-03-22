import argparse
import yaml
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Central Training Launcher for dLLM")
    
    # Core Config Paths
    parser.add_argument("--run_config", 
                        default="configs/train/lora_baseline.yaml",
                        help="Path to training/wandb configuration YAML")
    parser.add_argument("--slurm_config", 
                        default="configs/slurm/default.yaml",
                        help="Path to Slurm resource configuration YAML")
    parser.add_argument("--accelerate_config", 
                        default="fsdp", 
                        help="Name of accelerate config (located in configs/accelerate/)")
    
    # Collect remaining args to pass directly to training script
    args, extra_args = parser.parse_known_args()

    # Verify paths
    if not os.path.exists(args.run_config):
        print(f"Error: Run config not found at {args.run_config}")
        sys.exit(1)
    if not os.path.exists(args.slurm_config):
        print(f"Error: Slurm config not found at {args.slurm_config}")
        sys.exit(1)

    # 1. Load YAML Configurations
    with open(args.run_config, 'r') as f:
        run_cfg = yaml.safe_load(f)
    with open(args.slurm_config, 'r') as f:
        slurm_cfg = yaml.safe_load(f)

    # 2. Extract Slurm Directives
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

    # 3. Setup WandB and Environment Variables
    wb = run_cfg.get("wandb", {})
    env_exports = [
        f"export WANDB_NAME=\"{wb.get('name', 'unnamed')}\"",
        f"export WANDB_RUN_GROUP=\"{wb.get('group', 'default')}\"",
        f"export WANDB_TAGS=\"{','.join(wb.get('tags', []))}\"",
        f"export WANDB_PROJECT=\"{os.getenv('WANDB_PROJECT', 'BPTT-llada')}\"",
        "export WANDB_INIT_TIMEOUT=300",
        "export WANDB_DEBUG=false",
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
    ]

    # 4. Prepare Training Command
    training = run_cfg.get("training", {})
    script_path = training.pop("script_path", "examples/llada/sft.py")
    
    # 4a. Automatic Output Directory Construction from WandB
    if "output_dir" not in training:
        group = wb.get('group', 'default')
        name = wb.get('name', 'unnamed')
        # Format: .models/{group}/{name}
        training["output_dir"] = f".models/{group}/{name}"
        print(f"📂 Automatic output_dir: {training['output_dir']}")
    
    # 4b. Auto-resume logic
    if os.path.exists(training["output_dir"]) and any(d.startswith("checkpoint-") for d in os.listdir(training["output_dir"])):
        if "resume_from_checkpoint" not in training:
            training["resume_from_checkpoint"] = "True"
            print(f"🔄 Checkpoint found in {training['output_dir']}. Auto-resuming...")
    
    # Combine training params from YAML and CLI extra args
    train_flags = []
    for k, v in training.items():
        if isinstance(v, bool):
            if v: train_flags.append(f"--{k}")
        else:
            train_flags.append(f"--{k}")
            train_flags.append(str(v))
    train_flags.extend(extra_args)

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

module load slurm StdEnv/2023 python/3.11.5 cuda/12.6 cudnn
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
