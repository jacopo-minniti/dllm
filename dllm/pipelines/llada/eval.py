import os
import sys
from datetime import timedelta

# 1. FORCE initialization before any heavy imports to prevent early-default-init race conditions
if "RANK" in os.environ:
    try:
        import torch.distributed as dist
        import torch
        if not dist.is_initialized():
            # Sniff for the flag in raw sys.argv before official parsing
            timeout_seconds = 43200 
            for i, arg in enumerate(sys.argv):
                if arg == "--distributed_timeout" and i + 1 < len(sys.argv):
                    try:
                        timeout_seconds = int(sys.argv[i+1])
                    except ValueError:
                        pass
            
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo", 
                timeout=timedelta(seconds=timeout_seconds)
            )
    except ImportError:
        pass

# 2. Now proceed with the rest of the imports
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
    # The --distributed_timeout flag is now officially supported!
    try:
        cli_evaluate()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
