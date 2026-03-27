**Note**: README has been updated for our custom interventions and wiring!

# dLLM: Simple Diffusion Language Modeling

dLLM is a library for training and evaluating diffusion language models using aunified and scalable pipeline.

## 🚀 Quick Setup

All training and evaluation is designed to run via **Slurm**.

```bash
# 1. Initialize environment
conda create -n dllm python=3.11 -y
conda activate dllm

# 2. Install dependencies
pip install -e .
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Setup Evaluation Harness
git submodule update --init --recursive
pip install -e "lm-evaluation-harness"

# 4. Prepare logs director
mkdir .logs
```

## 🛠️ Training & Evaluation

We use a central launching system that combines YAML configurations with Slurm submission.

### Training
Configurations are stored in `configs/train/`.

```bash
# Launch a training run (e.g., LLaDA LoRA Baseline)
python train.py --run_config configs/train/lora_baseline.yaml
```

### Evaluation
Configurations are stored in `configs/eval/`. We use `lm-evaluation-harness` for benchmarks.

```bash
# Launch evaluation (e.g., MATH500)
python eval.py --run_config configs/eval/llada_math500.yaml
```

> [!TIP]
> Both scripts use `configs/slurm/default.yaml` for job resource requests (nodes, GPUs, time, etc.). You can also pass `--resume_from_checkpoint True` to `train.py` to pick up where a job left off.

## 📊 Monitoring (WandB)

Training progress is tracked via Weights & Biases. The system is wired with `WandbAlertCallback` to send Slack/Email alerts for run start, success, or failure. 

Ensure the following are in your environment:
```bash
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="BPTT-llada"
```

## 🧠 Custom LLaDA Implementation

Our implementation specialized LLaDA for improved convergence and stable instruction following:

- **PUMA (Probabilistic Unbiased Masking)**: Adjusts the masking distribution during training to prevent bias and ensure the model learns effectively at all noise levels.
- **2-Step BPTT (Backprop Through Time)**: Instead of standard single-step masking, we unroll the denoising process for 2 steps during training. This forces the model to learn trajectories that are stable over multi-step sampling.

### Code Organization
- `dllm/pipelines/llada/models/modeling_llada.py`: Core architecture, including PUMA and BPTT logic.
- `dllm/core/trainers/mdlm.py`: The `MDLMTrainer` which handles the diffusion loss calculations.
- `configs/`: Central YAML configurations for automated Slurm job generation.
- `train.py` / `eval.py`: Top-level launchers for the pipeline.
- `examples/llada/sft.py`: The supervised fine-tuning loop.
