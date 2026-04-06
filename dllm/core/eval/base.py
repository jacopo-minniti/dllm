"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import dataclasses
from dataclasses import dataclass
from datetime import timedelta

import accelerate
from accelerate.utils import InitProcessGroupKwargs
import torch
import torch.distributed as dist
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

import dllm
from dllm.core.samplers import BaseSampler, BaseSamplerConfig
from dllm.utils.configs import ModelArguments


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
        # Inference (especially diffusion) can be slow, increase timeout for stability.
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
        if "merge_lora" not in kwargs:
            kwargs["merge_lora"] = True
            
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = dllm.utils.get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = dllm.utils.get_tokenizer(self.model_args)
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

        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))

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
        # To handle uneven shards in distributed mode, we must synchronize the number of batches.
        num_local_requests = len(requests)
        if dist.is_initialized():
            max_req_tensor = torch.tensor([num_local_requests], device=self.device)
            dist.all_reduce(max_req_tensor, op=dist.ReduceOp.MAX)
            max_requests = max_req_tensor.item()
        else:
            max_requests = num_local_requests

        num_batches = (max_requests + self.batch_size - 1) // self.batch_size
        out: list[str] = [None] * num_local_requests

        for batch_idx in tqdm(
            range(num_batches), 
            desc="Generating...",
            disable=(self.rank != 0)
        ):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, num_local_requests)
            
            if start < num_local_requests:
                batch = requests[start:end]
                contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])
                prompts = [
                    torch.tensor(
                        self.tokenizer(ctx)["input_ids"],
                        device=self.device,
                        dtype=torch.long,
                    )
                    for ctx in contexts
                ]
            else:
                batch = []
                prompts = []

            # All ranks call sample() together to stay in sync.
            # MDLMSampler.sample is now robust to prompts=[].
            generated_ids = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=False,
            )
            
            if batch:
                generated_answers = dllm.utils.sample_trim(
                    self.tokenizer,
                    generated_ids.tolist(),
                    [p.tolist() for p in prompts],
                )

                for i, (answer, gen_kwargs) in enumerate(zip(generated_answers, gen_kwargs_list)):
                    for stop_seq in gen_kwargs["until"]:
                        if stop_seq in answer:
                            answer = answer.split(stop_seq)[0]
                    out[start + i] = answer

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather a tensor across ranks."""
        if dist.is_initialized():
            # Ensure tensor is on the correct device
            tensor = tensor.to(self.device)
            # lm-eval expects concatenated results
            output = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(output, tensor)
            return torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in output])
        return tensor

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        """Gather a Python object to all ranks (lm-eval typically only needs it on dst=0)."""
        if dist.is_initialized():
            output = [None] * self.world_size
            dist.all_gather_object(output, obj)
            return output
        return [obj]

    def barrier(self) -> None:
        """Synchronization barrier."""
        if dist.is_initialized():
            dist.barrier()
