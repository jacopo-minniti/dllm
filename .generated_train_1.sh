#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=03:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --output=./.logs/%x-%j.out
#SBATCH --error=./.logs/%x-%j.err
#SBATCH --requeue
#SBATCH --chdir=/scratch/jacopo04/dllm
#SBATCH --account=def-rudner
#SBATCH --signal=B:SIGTERM@360
#SBATCH --job-name=bptt-ctl_1
set -e
cd /scratch/jacopo04/dllm
module load StdEnv/2023 python/3.11.5 cuda/12.6 cudnn gcc opencv arrow
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -a
[ -f .env ] && . ./.env
set +a
export WANDB_NAME="lr2e-05_bs256_puma-th0.15_bptt2_cab-b512-e1024-rl17"
export WANDB_RUN_GROUP="LLaDA-8B-Base/unknown/base-puma-bptt-cab"
export WANDB_TAGS="LLaDA-8B-Base,bptt,cab,dllm,puma,unknown"
export WANDB_PROJECT="BPTT-llada"
export WANDB_INIT_TIMEOUT=300
export HF_HOME="$PWD/.cache"
export HF_DATASETS_CACHE="$PWD/.cache/datasets"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=43200
export ACCELERATE_TIMEOUT_IN_SECONDS=43200

MASTER_NAME=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist=$MASTER_NAME hostname -i | awk '{for(i=1;i<=NF;i++) if($i ~ /^10\.[0-9]+\./) {print $i; exit}}')
MASTER_ADDR=${MASTER_ADDR:-$MASTER_NAME}
MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 10000))

WORLD_SIZE=$((${SLURM_NNODES:-1} * ${SLURM_GPUS_ON_NODE:-1}))
LAUNCH_ARGS="--num_processes $WORLD_SIZE"
if [ "$WORLD_SIZE" -gt 1 ]; then
    LAUNCH_ARGS="--multi_gpu $LAUNCH_ARGS --num_machines ${SLURM_NNODES:-1}"
    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        LAUNCH_ARGS="$LAUNCH_ARGS --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --machine_rank \$SLURM_NODEID --rdzv_backend c10d"
    fi
    if [[ "configs/accelerate/fsdp.yaml" == *"fsdp"* ]]; then LAUNCH_ARGS="$LAUNCH_ARGS --fsdp_transformer_layer_cls_to_wrap LLaDABlock"; fi
fi

srun --label --ntasks-per-node=1 bash -c "accelerate launch $LAUNCH_ARGS --config_file "configs/accelerate/fsdp.yaml" "examples/llada/sft.py" --model_name_or_path GSAI-ML/LLaDA-8B-Base --bf16 --gradient_checkpointing --seed 42 --loss_type puma_bptt --distributed_timeout 43200 --num_train_epochs 1 --learning_rate 2e-05 --per_device_train_batch_size 8 --gradient_accumulation_steps 8 --lr_scheduler_type linear --warmup_ratio 0.1 --max_grad_norm 1.5 --dataset allenai/tulu-3-sft-mixture[train:50000,test:50000:52000] --logging_steps 10 --save_steps 100 --eval_steps 50 --use_puma --use_cab --use_bptt --bptt_steps 2 --cab_bottleneck_dim 512 --cab_mlp_expansion_dim 1024 --read_layers 17 --freeze_backbone --output_dir .models/LLaDA-8B-Base/unknown/base-puma-bptt-cab/lr2e-05_bs256_puma-th0.15_bptt2_cab-b512-e1024-rl17"
