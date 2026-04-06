"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot \
    --model llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0"
"""

import sys
import torch
from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.utils.utils import seed_everything


@dataclass
class LLaDAEvalSamplerConfig(MDLMSamplerConfig):
    """Default sampler config for LLaDA eval."""

    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class LLaDAEvalConfig(MDLMEvalConfig):
    """LLaDA eval config."""

    # According to LLaDA's opencompass implementation:
    # https://github.com/ML-GSAI/LLaDA/blob/main/opencompass/opencompass/models/dllm.py
    max_length: int = 4096


@register_model("llada")
class LLaDAEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: LLaDAEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or LLaDAEvalConfig()
        sampler_config = sampler_config or LLaDAEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    import os
    import datetime
    import torch.distributed as dist

    # Set aggressive NCCL and Torch environment variables for long-running jobs
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO") # For better debugging in case of failure

    # Ensure the process group is initialized with a very long timeout (12 hours)
    # This must happen as early as possible.
    if not dist.is_initialized():
        # In a slurm environment MASTER_ADDR and MASTER_PORT are usually set by srun or accelerate
        try:
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=datetime.timedelta(seconds=43200),
            )
        except Exception as e:
            print(f"Warning: Failed to initialize process group: {e}. Falling back to default.")

    # Pre-parse seed for process-wide determinism (cudnn, hashseed, etc.)
    eval_seed = 42
    for i, arg in enumerate(sys.argv):
        if arg == "--seed" and i + 1 < len(sys.argv):
            try:
                # Support both --seed 42 and --seed 42,42,42,42
                val = sys.argv[i+1].split(",")[0]
                if val.lower() != "none":
                    eval_seed = int(val)
            except (ValueError, IndexError):
                pass
            break
    
    seed_everything(eval_seed)
    try:
        cli_evaluate()
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
