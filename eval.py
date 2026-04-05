import argparse
import yaml
import subprocess
import os
import sys
from dllm.utils.naming import get_eval_naming

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
    parser.add_argument("--job_name", default="dllm", help="Slurm job name")
    
    # Collect remaining args to pass directly to the eval script
    args, extra_args = parser.parse_known_args()

    # Resolve paths
    args.run_config = resolve_config_path(args.run_config, "configs/eval")
    args.slurm_config = resolve_config_path(args.slurm_config, "configs/slurm")

    # 1. Load YAML Configurations
    with open(args.run_config, 'r') as f:
        try:
            run_cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            run_cfg = {}
    with open(args.slurm_config, 'r') as f:
        try:
            slurm_cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            slurm_cfg = {}
    
    # 1a. Set default job name or override from CLI
    slurm_cfg["job_name"] = args.job_name

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
    evaluation = run_cfg.get("evaluation", {})
    script_path = evaluation.pop("script_path", "dllm/pipelines/llada/eval.py")
    
    wb = evaluation.pop("wandb_args", {})
    env_exports = [
        f"export WANDB_NAME=\"{wb.get('name', 'eval')}\"",
        f"export WANDB_RUN_GROUP=\"{wb.get('group', 'evals')}\"",
        f"export WANDB_TAGS=\"{wb.get('tags', '').replace(',', '|')}\"", # Use pipe as a safe intermediate
        "export PYTHONUNBUFFERED=1",
        "export NCCL_DEBUG=INFO", 
        "export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1200"
    ]
    
    # 3b. Automatic Naming and Directory Construction
    if "seed" not in evaluation: evaluation["seed"] = 42
    
    # Extract model_args and handle CLI overrides
    model_args = evaluation.get("model_args", {})
    for arg in extra_args:
        if "--model_args" in arg:
            arg_content = arg.split("--model_args")[-1].strip().strip("\"'")
            for kv in arg_content.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    model_args[k.strip()] = v.strip()

    evaluation["model_args"] = model_args
    task_slug, model_slug, checkpoint_name, params_slug, _ = get_eval_naming(evaluation)
    
    # Configure results storage
    # lm-eval appends <model_name>/results.json automatically. 
    # To avoid duplicates, we provide a base path with all context except the model name.
    # Structure: .evals/<task>/<checkpoint>/<params>/<model>/results.json
    output_base = os.path.join(".evals", task_slug, checkpoint_name, params_slug)
    evaluation["output_path"] = output_base
    
    print(f"📦 Task: {task_slug}")
    print(f"📦 Model: {model_slug} ({checkpoint_name})")
    print(f"📦 Params: {params_slug}")
    print(f"📦 Results Base: {output_base}")

    # ── Determinism ──────────────────────────────────────────
    seed = evaluation.get("seed", 42)
    eval_flags = [f"--seed {seed}"]
    
    # Handle remaining dict-based args (model_args, gen_kwargs)
    # Re-inject updated model_args back into evaluation if needed
    evaluation["model_args"] = model_args
    for key in ["model_args", "gen_kwargs"]:
        val = evaluation.pop(key, None)
        if val:
            if isinstance(val, dict):
                val_str = dict_to_arg_str(val)
                eval_flags.append(f"--{key} \"{val_str}\"")
            else:
                eval_flags.append(f"--{key} \"{val}\"")
                
    # Default: force samples to stdout via write_out for the .out log
    if "write_out" not in evaluation:
        eval_flags.append("--write_out")

    # Add remaining flags
    for k, v in evaluation.items():
        if isinstance(v, bool):
            if v:
                eval_flags.append(f"--{k}")
        elif v is not None:
            eval_flags.append(f"--{k}")
            eval_flags.append(str(v))
    
    eval_flags.extend(extra_args)

    # Handle accelerate config path
    acc_config = args.accelerate_config
    if not acc_config.endswith(".yaml"):
        acc_config = f"configs/accelerate/{acc_config}.yaml"

    # Get working directory
    working_dir = slurm_cfg.get("working_dir", os.getcwd())

    eval_flags_escaped = [f.replace('"', '\\"') for f in eval_flags]
    
    # 4. Generate the Slurm Bash Script
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

{chr(10).join(env_exports)}

# ===== Scale Calculation =====
# Evaluation is forced to 1 node for stability
NUM_NODES=1
GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\\n' | wc -l)
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames "${{SLURM_JOB_NODELIST:-localhost}}" | head -n 1)
MASTER_PORT=$((20000 + ${{SLURM_JOB_ID:-0}} % 10000))

if [ -z "$SLURM_JOB_ID" ]; then
    MASTER_ADDR="localhost"
    MASTER_PORT=29500
    SLURM_PROCID=0
fi

echo "Launching Evaluation: NUM_NODES=$NUM_NODES, GPUS=$WORLD_SIZE on $MASTER_ADDR:$MASTER_PORT"

# ===== Execution =====
srun --ntasks-per-node=1 --nodes="${{NUM_NODES}}" \\
  bash -c "accelerate launch \\
  --timeout 43200 \\
  --config_file '{acc_config}' \\
  --num_machines '${{NUM_NODES}}' \\
  --num_processes '${{WORLD_SIZE}}' \\
  --main_process_ip '${{MASTER_ADDR}}' \\
  --main_process_port '${{MASTER_PORT}}' \\
  --rdzv_backend static \\
  --machine_rank \$SLURM_PROCID \\
  '{script_path}' {' '.join(eval_flags_escaped)}"
"""

    # Write to a temporary file
    temp_script = ".generated_eval.sh"
    with open(temp_script, "w") as f:
        f.write(bash_script)
    
    # Submit job
    os.makedirs(".logs", exist_ok=True)
    print(f"🚀 Submitting evaluation job via configuration: {args.run_config}")
    result = subprocess.run(["sbatch", temp_script])
    
    if result.returncode == 0:
        print(f"✅ Job submitted successfully. Script: {temp_script}")
    else:
        print(f"❌ Failed to submit job.")

if __name__ == "__main__":
    main()
