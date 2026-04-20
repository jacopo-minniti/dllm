# dLLM

Framework for training and evaluating diffusion language models. The primary focus is **Fast-dLLM** (a 1.5B block-diffusion model based on Qwen2.5), with additional support for LLaDA-8B.

Key training methods:
- **PUMA** — adjusts the masking distribution to remove sampling bias at all noise levels
- **BPTT** — unrolls 2 denoising steps during training so the model learns multi-step trajectories
- **Loopholing** — injects hidden-state memory from the previous denoising step into the next step's embeddings via a learned gated residual
- **CAB** — same idea as Loopholing but uses a cross-attention bridge instead of a linear projection

---

## Setup

```bash
conda create -n dllm python=3.11 -y && conda activate dllm
pip install -e .
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
git submodule update --init --recursive
pip install -e lm-evaluation-harness
mkdir .logs
```

Set your WandB credentials:

```bash
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="dllm"
```

---

## Running experiments

All experiments are launched through `train.py` and `eval.py`. Each script reads a YAML config, resolves keyword shortcuts from `configs/keywords.yaml`, and submits a Slurm job.

```bash
python train.py --run_config <config> [--slurm_config <slurm>]
python eval.py  --run_config <config> [--slurm_config <slurm>]
```

The `--slurm_config` argument selects a resource profile from `configs/slurm/`. Available profiles: `h100_high` (8×H100, 3h), `h100_mid`, `h100_low`, `l40_mid`, `test`. Defaults to `h100_high` if omitted.

To resume training from the latest checkpoint, pass `--resume_from_checkpoint True` to `train.py`.

### Training configs (`configs/train/`)

| Config | Model | Method | Notes |
|--------|-------|--------|-------|
| `base.yaml` | Fast-dLLM | MLM baseline | Full backbone, no PUMA/BPTT |
| `fast_loophole.yaml` | Fast-dLLM | Loopholing + PUMA + BPTT (LoRA) | Main method |
| `cab_fast.yaml` | Fast-dLLM | CAB + PUMA + BPTT (LoRA) | Cross-attention bridge variant |
| `cab.yaml` | Fast-dLLM | CAB + PUMA + BPTT (frozen backbone) | Adapter-only training |
| `puma_loopholing.yaml` | LLaDA-8B | Loopholing + BPTT | LLaDA baseline |
| `bptt_control.yaml` | LLaDA-8B | BPTT only | Ablation: BPTT without Loopholing |

**Example — train the main Fast-dLLM Loopholing experiment on 8×H100:**

```bash
python train.py --run_config configs/train/fast_loophole.yaml --slurm_config h100_high
```

### Evaluation configs (`configs/eval/`)

| Config | What it evaluates |
|--------|-------------------|
| `fast_base.yaml` | Fast-dLLM pretrained baseline on MATH500 |
| `fast_loophole.yaml` | Trained Loopholing checkpoint on MATH500 |
| `cab.yaml` | Trained CAB checkpoint on MATH500 |
| `puma_loopholing.yaml` | LLaDA-8B Loopholing checkpoint |

**Example:**

```bash
python eval.py --run_config configs/eval/fast_base.yaml --slurm_config h100_mid
```

---

## Config system

`configs/keywords.yaml` maps shorthand keys to full parameter sets. For example:

```yaml
mode: loophole_bptt   # expands to use_puma=True, use_loopholing=True, use_bptt=True, loss_type=puma_bptt
backbone: lora         # expands to freeze_backbone=False, lora=True, r=64, lora_alpha=128
bptt: 2               # expands to bptt_steps=2
```

`configs/default.yaml` sets base defaults that are merged into every config.

---

## Code layout

```
train.py / eval.py              Top-level launchers (config resolution + Slurm submission)
examples/
  llada/sft.py                  LLaDA training entry point
  fastdllm_v2/sft.py            Fast-dLLM training entry point
dllm/core/
  trainers/mdlm.py              MDLMTrainer — HuggingFace Trainer subclass
  losses.py                     MLMLoss, PumaLoss, LoopholingBPTTLoss, LoopholingBPTTPumaLoss
  samplers/mdlm.py              MDLMSampler — iterative confidence-based unmasking
  schedulers/                   Noise alpha schedulers
dllm/data/
  streaming_batch.py            StreamingBatch — on-policy buffer with row-wise eviction
dllm/pipelines/
  llada/                        LLaDA-8B model + eval pipeline
  fastdllm_v2/                  Fast-dLLM model + eval pipeline
configs/
  train/ eval/ slurm/           Experiment, evaluation, and resource configs
  keywords.yaml                 Shorthand key → parameter expansion
  default.yaml                  Base defaults merged into every config
```