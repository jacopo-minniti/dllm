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
from dataclasses import dataclass

import accelerate
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
        accelerator = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
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
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        batch_size = kwargs.get("batch_size", eval_config.batch_size)
        if batch_size == "auto":
            print("Warning: batch_size='auto' not yet supported for dLLM custom harness. Falling back to 1.")
            self.batch_size = 1
        else:
            self.batch_size = int(batch_size)

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

    # ── Evaluation Logic ──────────────────────────────────────────────
    
    def _truncate_hallucinations(self, text: str) -> str:
        """
        Base models hallucinate subsequent few-shot examples. 
        This cuts off the generation when a new question starts.
        """
        if not text:
            return ""
        
        return text[:20]

    def _normalize(self, text: str) -> str:
        """
        Standardizes math strings by removing formatting and spaces.
        """
        if not text:
            return ""
        
        text = str(text).lower()
        text = text.replace(r"\\", "").replace("\\", "")
        text = text.replace("{", "").replace("}", "")
        text = text.replace("x=", "").replace("y=", "").replace("z=", "")
        text = re.sub(r"\s+", "", text)
        
        return text

    def _is_strict_match(self, norm_solution: str, norm_answer: str) -> tuple[bool, any]:
        """
        Checks if the solution is in the answer with dynamic boundaries.
        """
        if not norm_solution or not norm_answer:
            return False, None

        escaped_sol = re.escape(norm_solution)

        # Dynamic boundaries
        prefix_bound = r"(?<!\d)" if norm_solution[0].isdigit() else r"(?<![a-z])" if norm_solution[0].isalpha() else ""
        suffix_bound = r"(?!\d)" if norm_solution[-1].isdigit() else r"(?![a-z])" if norm_solution[-1].isalpha() else ""

        pattern = prefix_bound + escaped_sol + suffix_bound
        match = re.search(pattern, norm_answer)
        
        return match is not None, match

    def _extract_raw_snippet(self, truncated_answer: str, norm_solution: str, context_size: int = 150) -> str:
        """
        Reverse-engineers the normalization to find the exact unnormalized snippet.
        """
        if not norm_solution:
            return truncated_answer[-context_size:] if len(truncated_answer) > context_size else truncated_answer

        pattern_chars = [re.escape(char) for char in norm_solution]
        filler = r'[\s\\{}]*'
        pattern = filler.join(pattern_chars)
        
        match = re.search(pattern, str(truncated_answer).lower())
        if match:
            start, end = match.span()
            start_idx = max(0, start - context_size)
            end_idx = min(len(truncated_answer), end + context_size)
            snippet = truncated_answer[start_idx:end_idx]
            return f"...{snippet}..."
        
        return truncated_answer[-context_size:] if len(truncated_answer) > context_size else truncated_answer

    # ── Unified generate_until scaffolding ────────────────────────────
    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        # Track if we have a checkpoint path to resume from
        checkpoint_path = getattr(self.model_args, "eval_checkpoint", None)
        processed_results = {}
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"🔄 Resuming evaluation from checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Support both new 'question' and old 'context' keys for resume
                        ctx = data.get("question", data.get("context"))
                        if ctx:
                            processed_results[ctx] = data["answer"]
                    except Exception:
                        continue
            print(f"✅ Loaded {len(processed_results)} existing results.")

        out: list[str] = [None] * len(requests)
        pending_indices = []
        
        for i, inst in enumerate(requests):
            ctx = inst.args[0]
            if ctx in processed_results:
                out[i] = processed_results[ctx]
            else:
                pending_indices.append(i)
        
        if not pending_indices:
            return out

        # Process only pending requests
        pending_requests = [requests[i] for i in pending_indices]
        
        for batch_start in tqdm(
            range(0, len(pending_requests), self.batch_size), desc="Generating (Pending)..."
        ):
            batch_idxs = pending_indices[batch_start : batch_start + self.batch_size]
            batch = [requests[i] for i in batch_idxs]
            contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

            prompts = [
                torch.tensor(
                    self.tokenizer(ctx)["input_ids"],
                    device=self.device,
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

            # Post-process and checkpoint
            new_saves = []
            for j, (answer, gen_kwargs) in enumerate(zip(generated_answers, gen_kwargs_list)):
                for stop_seq in gen_kwargs["until"]:
                    if stop_seq in answer:
                        answer = answer.split(stop_seq)[0]
                
                real_idx = batch_idxs[j]
                out[real_idx] = answer
                
                if checkpoint_path:
                    # Capture richer information if available
                    real_idx = batch_idxs[j]
                    inst = requests[real_idx]
                    
                    # Extract ground truth solution
                    doc = getattr(inst, "doc", None)
                    raw_solution = ""
                    if doc:
                        for key in ["answer", "gold", "target", "solution"]:
                            if key in doc:
                                raw_solution = doc[key]
                                break
                    
                    # 1. Truncate
                    truncated_answer = self._truncate_hallucinations(answer)
                    
                    # 2. Normalize
                    norm_answer = self._normalize(truncated_answer)
                    norm_solution = self._normalize(raw_solution)
                    
                    # 3. Match
                    is_correct, _ = self._is_strict_match(norm_solution, norm_answer)
                    
                    if is_correct:
                        if not hasattr(self, "_printed_correct_count"):
                            self._printed_correct_count = {}
                        task_name = getattr(inst, "task_name", "eval")
                        count = self._printed_correct_count.get(task_name, 0)
                        
                        if count < 20:
                            snippet = self._extract_raw_snippet(truncated_answer, norm_solution, context_size=150)
                            print(f"Row {real_idx} | ✅ CORRECT (Match #{count + 1})")
                            print(f"  Raw Solution: {raw_solution}")
                            print(f"  Context:\n  {snippet.strip()}")
                            print("-" * 60)
                            self._printed_correct_count[task_name] = count + 1
                    
                    # Reorder fields based on requirements:
                    # first the ground truth solution, then the is_correct, 
                    # then the question and finally the answer genrated by the model.
                    save_record = {
                        "solution": raw_solution,
                        "is_correct": bool(is_correct),
                        "question": contexts[j], 
                        "answer": answer,
                    }
                    
                    # Extract task name for logging if needed
                    task_name = getattr(inst, "task_name", None)
                    metadata = getattr(inst, "metadata", None)
                    if not task_name and isinstance(metadata, (list, tuple)) and len(metadata) > 0:
                        task_name = metadata[0]
                    if task_name:
                        save_record["task"] = task_name

                    new_saves.append(save_record)

            if checkpoint_path and new_saves:
                with open(checkpoint_path, "a") as f:
                    for item in new_saves:
                        f.write(json.dumps(item) + "\n")

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
