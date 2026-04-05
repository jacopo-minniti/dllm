"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import os
import dataclasses
import json
import re
import traceback
from datetime import timedelta
from dataclasses import dataclass

import accelerate
from accelerate.utils import InitProcessGroupKwargs
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks import get_task_dict
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
        # Inference (especially diffusion) can be extremely slow per batch.
        # We increase the NCCL timeout to 4 hours to prevent "Straggler" timeout errors.
        timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
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
        # Track if we have a checkpoint path to resume from
        checkpoint_path = getattr(self.model_args, "eval_checkpoint", None)
        processed_results = {} # ctx -> answer
        processed_questions = {} # clean_q -> answer
        
        def _extract_q(c):
            """Helper to extract the main question from a prompt, skipping few-shots."""
            for delimiter in ["Problem:", "Question:", "Input:", "Q:"]:
                if delimiter in c:
                    return c.split(delimiter)[-1].strip()
            return c.strip()

        # ── Loading Checkpoint ──────────────────────────────────
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self.rank == 0:
                print(f"🔄 Resuming evaluation from checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        ctx = data.get("prompt", data.get("question", data.get("context")))
                        ans = data.get("answer")
                        if ctx and ans is not None:
                            processed_results[ctx] = ans
                            processed_questions[_extract_q(ctx)] = ans
                    except Exception:
                        continue
            if self.rank == 0:
                print(f"✅ Loaded {len(processed_results)} cached results.")

        # ── Synchronized Pending Indices ──────────────────────────
        # All ranks must agree on which indices are pending to avoid division mismatches.
        pending_indices = []
        if self.rank == 0:
            for i, inst in enumerate(requests):
                ctx = inst.args[0]
                q_text = _extract_q(ctx)
                if ctx not in processed_results and q_text not in processed_questions:
                    pending_indices.append(i)
        
        if self.world_size > 1:
            pending_indices = self.gather_object(pending_indices)[0]

        # Fill 'out' list with whatever we already have
        out: list[str] = [None] * len(requests)
        for i, inst in enumerate(requests):
            if i not in pending_indices:
                ctx = inst.args[0]
                q_text = _extract_q(ctx)
                out[i] = processed_results.get(ctx, processed_questions.get(q_text))

        if self.rank == 0:
            print(f"📊 Total: {len(requests)} | Cached: {len(requests) - len(pending_indices)} | Pending: {len(pending_indices)}")

        # ── Distributed Splitting ────────────────────────────────
        if self.world_size > 1:
            rank_pending_indices = [idx for idx in pending_indices if idx % self.world_size == self.rank]
            if self.rank == 0:
                print(f"🌐 Distributed Eval: Rank {self.rank} processing {len(rank_pending_indices)}/{len(pending_indices)} pending samples.")
        else:
            rank_pending_indices = pending_indices

        if rank_pending_indices:
            pending_requests = [requests[i] for i in rank_pending_indices]
            
            for batch_start in tqdm(
                range(0, len(pending_requests), self.batch_size), 
                desc=f"Generating (Rank {self.rank})",
                disable=(self.rank != 0 and len(pending_requests) < 50)
            ):
                try:
                    batch_idxs = rank_pending_indices[batch_start : batch_start + self.batch_size]
                    batch = [requests[i] for i in batch_idxs]
                    contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

                    prompts = [
                        torch.tensor(
                            self.tokenizer(ctx)["input_ids"],
                            device=self._device,
                            dtype=torch.long,
                        )
                        for ctx in contexts
                    ]

                    generated_ids = self.sampler.sample(
                        inputs=prompts,
                        config=self.sampler_config,
                        return_dict=False,
                    )
                    generated_answers = sample_trim(
                        self.tokenizer,
                        generated_ids.tolist(),
                        [p.tolist() for p in prompts],
                    )

                    new_saves = []
                    for j, (answer, gen_kwargs) in enumerate(zip(generated_answers, gen_kwargs_list)):
                        for stop_seq in gen_kwargs["until"]:
                            if stop_seq in answer:
                                answer = answer.split(stop_seq)[0]
                        
                        real_idx = batch_idxs[j]
                        out[real_idx] = answer
                        
                        if checkpoint_path:
                            inst = requests[real_idx]
                            doc = getattr(inst, "doc", None)
                            raw_solution = ""
                            if doc:
                                for key in ["answer", "gold", "target", "solution"]:
                                    if key in doc:
                                        raw_solution = doc[key]
                                        break
                            
                            task_name = getattr(inst, "task_name", None)
                            if not task_name and isinstance(inst.metadata, (list, tuple)) and len(inst.metadata) > 0:
                                task_name = inst.metadata[0]
                            
                            is_correct = False
                            if task_name:
                                if not hasattr(self, "_task_objects"):
                                    self._task_objects = {}
                                if task_name not in self._task_objects:
                                    try:
                                        self._task_objects[task_name] = get_task_dict([task_name])[task_name]
                                    except Exception as e:
                                        if self.rank == 0: print(f"⚠️ Error loading task {task_name}: {e}")
                                        self._task_objects[task_name] = None
                                
                                task = self._task_objects.get(task_name)
                                if task:
                                    try:
                                        res = task.process_results(doc, [answer])
                                        is_correct = any(bool(v == 1.0 or v is True) for v in res.values())
                                    except Exception as e:
                                        if self.rank == 0: print(f"⚠️ Error processing row {real_idx}: {e}")
                            
                            if is_correct and self.rank == 0:
                                if not hasattr(self, "_printed"): self._printed = 0
                                if self._printed < 5:
                                    print(f"Row {real_idx} | ✅ CORRECT | Sol: {raw_solution}")
                                    self._printed += 1
                            
                            save_record = {
                                "solution": raw_solution,
                                "is_correct": bool(is_correct),
                                "question": _extract_q(contexts[j]),
                                "prompt": contexts[j],
                                "answer": answer,
                            }
                            if task_name: save_record["task"] = task_name
                            new_saves.append(save_record)

                    if checkpoint_path and new_saves:
                        with open(checkpoint_path, "a") as f:
                            for item in new_saves:
                                f.write(json.dumps(item) + "\n")

                except Exception as e:
                    if self.rank == 0:
                        print(f"❌ Error in batch {batch_start}: {e}")
                        traceback.print_exc()
                    continue

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
