"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import os
import dataclasses
import traceback
from datetime import timedelta
from dataclasses import dataclass

import accelerate
from accelerate.utils import InitProcessGroupKwargs
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

from ..samplers import BaseSampler, BaseSamplerConfig
from ...utils import (
    get_model,
    get_tokenizer,
    sample_trim,
)
from ...utils.configs import ModelArguments


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: device and batch_size."""

    pretrained: str = ""
    device: str = "cuda"
    batch_size: int = 1

    def get_model_config(self, pretrained: str):
        """Optional: return custom model config for loading. Default None (use checkpoint config)."""
        return None


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood (and optionally loglikelihood_rolling);
    generate_until is implemented here and uses sampler + sampler_config.
    """

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        """Build a dataclass *config_cls* by copying fields from *source*, with *kwargs* overrides."""
        init = {}
        for f in dataclasses.fields(config_cls):
            if f.name in kwargs:
                init[f.name] = kwargs[f.name]
            elif hasattr(source, f.name):
                init[f.name] = getattr(source, f.name)
        return config_cls(**init)

    def __init__(
        self,
        eval_config: BaseEvalConfig | None = None,
        model_args: ModelArguments | None = None,
        sampler_config: BaseSamplerConfig | None = None,
        sampler_cls: type[BaseSampler] | None = None,
        **kwargs,
    ):
        super().__init__()
        eval_config = eval_config or BaseEvalConfig()
        # Ensure model path is in kwargs and we have a safe default for ModelArguments(__post_init__).
        model_args = model_args or ModelArguments(
            model_name_or_path=kwargs.get("pretrained")
        )
        device = kwargs.get("device", eval_config.device)

        # ── Distributed ──────────────────────────────────────────
        # Inference (especially diffusion) can be extremely slow per batch (~150s/it).
        # We increase the NCCL timeout to 12 hours (43200s) to prevent "Straggler" timeout errors.
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=43200))
        accelerator = accelerate.Accelerator(kwargs_handlers=[timeout_kwargs])
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
        
        # Default to merging LoRA for eval speed unless explicitly disabled
        if "merge_lora" not in kwargs and (model_args is None or not hasattr(model_args, "merge_lora")):
            kwargs["merge_lora"] = True
            
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = get_tokenizer(self.model_args)
        if sampler_config is not None:
            self.sampler_config = self._build_config(
                type(sampler_config), sampler_config, kwargs
            )
        if sampler_cls is not None:
            self.sampler = sampler_cls(model=self.model, tokenizer=self.tokenizer)

        # ── Device placement ─────────────────────────────────────
        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self._device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self._device = torch.device(device)
            self.accelerator = None

        batch_size = kwargs.get("batch_size", eval_config.batch_size)
        if batch_size == "auto":
            print("Warning: batch_size='auto' not yet supported for dLLM custom harness. Falling back to 1.")
            self.batch_size = 1
        else:
            self.batch_size = int(batch_size)

    def all_gather(self, tensor):
        """All-gather a tensor across ranks using accelerate."""
        if self.accelerator is not None:
            return self.accelerator.gather(tensor)
        return tensor

    def gather_object(self, obj, dst=0):
        """Gather a Python object to dst rank using dist.all_gather_object."""
        import torch.distributed as dist
        if dist.is_initialized():
            output = [None for _ in range(self.world_size)]
            dist.all_gather_object(output, obj)
            return output
        return [obj]

    def barrier(self):
        """Synchronization barrier using accelerate."""
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format chat history for input to the LM."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    # ── Unified generate_until scaffolding ────────────────────────────
    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        # ── Distributed Setup ─────────────────────────────────────
        out: list[str] = [None] * len(requests)
        
        # Determine number of batches needed
        num_total_batches = (len(requests) + (self.world_size * self.batch_size) - 1) // (self.world_size * self.batch_size)
        
        # Everyone iterates the same number of times to keep ranks alive
        from tqdm import tqdm
        import sys
        
        for batch_num in tqdm(
            range(num_total_batches),
            desc=f"Generating",
            disable=(self.rank != 0),
            file=sys.stdout,
            mininterval=1.0,
            maxinterval=5.0
        ):
            # Calculate which global indices are assigned to THIS rank for THIS batch
            sub_indices = []
            start_off = batch_num * self.world_size * self.batch_size
            for i in range(self.batch_size):
                global_idx_in_batch = start_off + (i * self.world_size) + self.rank
                if global_idx_in_batch < len(requests):
                    sub_indices.append(global_idx_in_batch)
            
            try:
                # IMPORTANT: All ranks MUST call the sampler every iteration, even if they have 0 samples.
                batch_requests = [requests[i] for i in sub_indices] if sub_indices else []
                contexts = [inst.args[0] for inst in batch_requests]
                prompts = [
                    torch.tensor(self.tokenizer(ctx)["input_ids"], device=self._device, dtype=torch.long)
                    for ctx in contexts
                ]

                # Distributed sync point (implicit if model is wrapped in FSDP/DDP)
                generated_ids = self.sampler.sample(
                    inputs=prompts,
                    config=self.sampler_config,
                    return_dict=False,
                )

                if sub_indices:
                    _, gen_kwargs_list = zip(*[inst.args for inst in batch_requests])
                    generated_answers = sample_trim(
                        self.tokenizer,
                        generated_ids.tolist(),
                        [p.tolist() for p in prompts],
                    )

                    for j, (answer, gen_kwargs) in enumerate(zip(generated_answers, gen_kwargs_list)):
                        for stop_seq in gen_kwargs["until"]:
                            if stop_seq in answer:
                                answer = answer.split(stop_seq)[0]
                        
                        real_idx = sub_indices[j]
                        out[real_idx] = answer

            except Exception as e:
                # Print the error regardless of the rank so you can diagnose the real issue
                print(f"❌ Error on Rank {self.rank} in batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                
                # Raise the exception to kill the job instantly instead of hanging the other GPUs
                raise e

        # ── Final Result Synchronization ─────────────────────────
        if self.world_size > 1:
            if self.rank == 0:
                print("🔄 Synchronizing results across ranks...")
            
            all_rank_outs = self.gather_object(out)
            final_out = [None] * len(requests)
            for rank_out in all_rank_outs:
                if rank_out is None: continue
                for i, ans in enumerate(rank_out):
                    if ans is not None:
                        final_out[i] = ans
            
            missing = sum(1 for x in final_out if x is None)
            if missing > 0 and self.rank == 0:
                print(f"⚠️ Warning: {missing}/{len(requests)} requests missing results.")
            
            return final_out

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
