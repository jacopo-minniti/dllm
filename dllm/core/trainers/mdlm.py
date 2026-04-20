"""
References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState

from ..schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.configs import TrainingArguments
from dllm.utils.data import prepend_bos
from .utils import NLLMetric, PPLMetric, StatsMetric, OnEvaluateMetricsCallback, WandbAlertCallback, SlurmCheckpointCallback, ModelingFilesSyncCallback
from ..losses import MLMLoss, PumaLoss, LoopholingBPTTLoss, LoopholingBPTTPumaLoss
from dllm.data.streaming_batch import StreamingBatch


@dataclass
class MDLMConfig(TrainingArguments):
    time_epsilon: float = 1e-3
    loss_weight_type: str = "scheduler"  # "scheduler", "uniform"
    loss_norm_type: str = "token"  # "batch", "sequence", "token"
    right_shift_logits: bool = False
    loss_type: str = "mlm"  # "mlm", "puma", "bptt", "puma_bptt"
    puma_threshold: float = 0.15
    bptt_steps: int = 2
    confidence_type: str = "top_prob" # "top_prob", "prob_diff"
    weighted_ce: bool = False


class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        args: MDLMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        processing_class: Any | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, processing_class=processing_class, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")
        if args.confidence_type not in {"top_prob", "prob_diff"}:
            raise ValueError(
                f"Unsupported confidence_type={args.confidence_type!r}. "
                "Supported values: top_prob, prob_diff."
            )

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits

        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={
                "nll": NLLMetric(), 
                "ppl": PPLMetric(),
                "intervention_ratio": StatsMetric(),
                "gamma_mean": StatsMetric(),
                "h_s_norm": StatsMetric(),
                "h_t_norm": StatsMetric(),
            },
        )
        self.add_callback(self.meter)
        self.add_callback(WandbAlertCallback())
        self.add_callback(SlurmCheckpointCallback())

        # Ensure every checkpoint directory is self-contained by copying the pipeline's
        # modeling .py files alongside the weights at save time (training time responsibility).
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type:
            self.add_callback(ModelingFilesSyncCallback(model_type))

        # Registry of loss functions
        self.loss_fns = {
            "mlm": MLMLoss(self.processing_class.mask_token_id),
            "puma": PumaLoss(
                self.processing_class.mask_token_id, 
                threshold=args.puma_threshold, 
                confidence_type=args.confidence_type
            ),
            "bptt": LoopholingBPTTLoss(
                self.processing_class.mask_token_id, 
                num_steps=args.bptt_steps
            ),
            "puma_bptt": LoopholingBPTTPumaLoss(
                self.processing_class.mask_token_id, 
                threshold=args.puma_threshold, 
                num_steps=args.bptt_steps,
                confidence_type=args.confidence_type,
                weighted_ce=args.weighted_ce
            ),
        }
        self.loss_fn = self.loss_fns.get(args.loss_type, self.loss_fns["mlm"])
        self.active_streaming_batch = None
        # Ensure loopholing is enabled on the model if needed by the loss function
        if "bptt" in args.loss_type:
             use_state = False
             model_config = None
             if hasattr(self.model, "config"):
                  model_config = self.model.config
             elif hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                  model_config = self.model.model.config
             
             if model_config is not None:
                  use_state = getattr(model_config, "use_loopholing", False) or getattr(model_config, "use_cab", False)
             
             if not use_state:
                  raise ValueError(
                      f"Loss function '{args.loss_type}' requires state persistence support (Loopholing or CAB). "
                      "Please enable one of them by adding '--use_loopholing True' or '--use_cab True' to your launch command."
                  )

        # Check for potential NCCL timeouts in distributed training with PUMA
        if (
            self.args.world_size > 1 
            and "puma" in self.args.loss_type 
            and self.args.max_steps < 0
        ):
             transformers.logging.get_logger(__name__).warning(
                 "You are running PUMA in a distributed environment without setting `max_steps`. "
                 "This may lead to NCCL timeouts if dataset shards have different lengths because "
                 "PUMA unmasking variability can cause different ranks to reach the end of their epochs "
                 "at slightly different times. Consider setting `max_steps` and ensuring your dataset is infinite."
             )

    def _wrap_model(self, model, training=True, dataloader=None):
        # BPTT unrolls num_steps forward passes through the same parameters and
        # sums the losses before a single backward(). DDP fires its gradient-ready
        # hook once per forward, so each parameter is marked ready num_steps times,
        # which DDP rejects.
        #
        # _set_static_graph() was the original fix but it requires the set of
        # *unused* parameters to be identical across all iterations. With LoRA +
        # PUMA the per-iteration participation varies (e.g. a slot with no masked
        # tokens never enters a LoRA block), so _set_static_graph() raises:
        #   "Expected to have finished reduction in the prior iteration before
        #    starting a new one."
        #
        # The correct fix is find_unused_parameters=True, set BEFORE DDP creation
        # (Trainer reads this flag inside super()._wrap_model). This lets DDP
        # dynamically discover unused params each iteration at a small overhead
        # cost — acceptable for BPTT workloads.
        #
        # Reentrant gradient checkpointing compounds the DDP hook issue: each
        # checkpointed segment re-runs its forward during backward, re-triggering
        # hooks. Non-reentrant checkpointing avoids that second trigger.
        if "bptt" in self.args.loss_type and training:
            if self.args.ddp_find_unused_parameters is None:
                self.args.ddp_find_unused_parameters = True

        wrapped = super()._wrap_model(model, training=training, dataloader=dataloader)

        if "bptt" in self.args.loss_type:
            # Switch the underlying model to non-reentrant checkpointing
            inner = wrapped.module if hasattr(wrapped, "module") else wrapped
            if hasattr(inner, "gradient_checkpointing_enable"):
                inner.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        return wrapped

    def save_model(self, output_dir: str | None = None, **kwargs):
        """Save model weights then copy pipeline modeling .py files into the same directory.

        The callback covers intermediate checkpoints (on_save).  This override covers the
        explicit trainer.save_model("checkpoint-final") call at the end of sft.py, which
        goes through save_model() directly and never triggers on_save.
        """
        super().save_model(output_dir=output_dir, **kwargs)
        target = output_dir or self.args.output_dir
        if target and PartialState().is_main_process:
            model_type = getattr(getattr(self.model, "config", None), "model_type", None)
            if model_type:
                from dllm.utils.models import sync_modeling_files
                sync_modeling_files(model_type, target)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)

            # If labels exist and EVERY sequence already starts with -100,
            # we treat them as is and skip prepending BOS.
            if labels is not None:
                # shape: [bsz, seq_len]
                if torch.all(labels[:, 0] == -100):
                    return inputs

            # Otherwise, prepend BOS (and corresponding labels / attention_mask).
            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "uniform":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the masked diffusion language modeling loss.

        Applies stochastic masking to input tokens based on a diffusion timestep,
        then computes the weighted cross-entropy loss for predicting the original tokens.

        Args:
            model: The language model to train.
            inputs: Dictionary containing input_ids, labels, and optionally attention_mask.
            return_outputs: If True, return both loss and model outputs.

        Returns:
            Loss tensor, or tuple of (loss, outputs) if return_outputs is True.
        """
        if self.args.loss_type == "mlm":
            # === Original MLM Logic (preserved for clean ablation) ===
            inputs = self._preprocess_inputs(inputs)
            input_ids, labels, attention_mask = (
                inputs["input_ids"],
                inputs["labels"],
                inputs.get("attention_mask", None),
            )
            b, l = input_ids.shape
            maskable_mask = labels != -100

            t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
                b, device=input_ids.device
            )
            p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)

            masked_mask = (
                torch.rand((b, l), device=input_ids.device) < p_mask
            ) & maskable_mask
            noised_input_ids = torch.where(
                masked_mask, self.processing_class.mask_token_id, input_ids
            )

            outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            token_nll = F.cross_entropy(
                logits.transpose(1, 2),
                input_ids,
                reduction="none",
            ).detach()
            
            self.meter.update(
                split="train" if model.training else "eval",
                value=token_nll * masked_mask.to(token_nll.dtype),
                weight=maskable_mask.to(dtype=logits.dtype).detach(),
            )

            # Update additional stats during evaluation
            if not model.training and hasattr(outputs, "stats"):
                for k, v in outputs.stats.items():
                    if v is not None:
                        self.meter.update_metric(split="eval", name=k, value=v)
            elif not model.training:
                # Direct metrics from model output (for 1-step MLM)
                for k in ["intervention_ratio", "gamma_mean"]:
                    v = getattr(outputs, k, None)
                    if v is not None:
                        self.meter.update_metric(split="eval", name=k, value=v)
                # h_s norm (h_t is None or Noise in MLM)
                if hasattr(outputs, "h_s") and outputs.h_s is not None:
                    self.meter.update_metric(split="eval", name="h_s_norm", value=outputs.h_s.norm(p=2, dim=-1).mean().item())

            # We need the non-detached version for the loss
            token_nll_orig = F.cross_entropy(
                logits.transpose(1, 2),
                input_ids,
                reduction="none",
            )
            
            loss_weights = self._compute_loss_weights(
                t=t, inputs=inputs, masked_mask=masked_mask
            ) 
            
            if self.loss_norm_type == "token":
                loss = (token_nll_orig * loss_weights * masked_mask.to(token_nll_orig.dtype)).sum() / maskable_mask.sum().clamp_min(1)
            elif self.loss_norm_type == "sequence":
                loss = ((token_nll_orig * loss_weights * masked_mask.to(token_nll_orig.dtype)).sum(-1) / maskable_mask.sum(-1).clamp_min(1)).mean()
            elif self.loss_norm_type == "batch":
                loss = (token_nll_orig * loss_weights * masked_mask.to(token_nll_orig.dtype)).sum() / b
            return (loss, outputs) if return_outputs else loss

        else:
            # === New Interventions (PUMA / BPTT) ===
            if "puma" in self.args.loss_type:
                # Use persistent row-wise streaming batch
                if self.active_streaming_batch is None:
                    self.active_streaming_batch = StreamingBatch()

                # Evict finished rows and fill with current dataloader input
                # Capacity should be batch_size * gradient_accumulation_steps to avoid discarding data
                capacity = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
                inputs = self._preprocess_inputs(inputs)

                # --- DEBUG: incoming dataloader batch ---
                from dllm.core.losses import _BPTT_DEBUG_STEP
                _do_debug = ((_BPTT_DEBUG_STEP + 1) % 10 == 1)
                if _do_debug:
                    _dl_ids = inputs["input_ids"]
                    _dl_labs = inputs["labels"]
                    _dl_maskable = (_dl_labs != -100).sum().item()
                    _dl_has_mask = (_dl_ids == self.processing_class.mask_token_id).sum().item()
                    print(
                        f"\n[TRAINER DEBUG] Incoming batch shape={tuple(_dl_ids.shape)}  "
                        f"maskable_tokens={_dl_maskable}  "
                        f"already_masked={_dl_has_mask}  "
                        f"streaming_buf={'None' if self.active_streaming_batch.input_ids is None else f'capacity={self.active_streaming_batch.capacity} ready_to_evict={self.active_streaming_batch.ready_to_evict.sum().item() if self.active_streaming_batch.ready_to_evict is not None else None}'}",
                        flush=True,
                    )

                # Filled slots from current mini-batch
                slots = self.active_streaming_batch.evict_and_fill(
                    inputs,
                    self.processing_class.mask_token_id,
                    capacity=capacity
                )

                if _do_debug:
                    _buf_masked = (self.active_streaming_batch.input_ids == self.processing_class.mask_token_id).sum().item()
                    _buf_maskable = (self.active_streaming_batch.labels != -100).sum().item()
                    print(
                        f"[TRAINER DEBUG] After evict_and_fill: "
                        f"filled_slots={slots.tolist()}  "
                        f"buf_total_masked={_buf_masked}  "
                        f"buf_total_maskable={_buf_maskable}",
                        flush=True,
                    )

                # If no slots were ready to evict, we still must process 'batch_size' sequences
                # to maintain training throughput and memory usage matching baseline.
                if len(slots) < self.args.per_device_train_batch_size:
                    # Pick remaining random slots from the buffer that are NOT ready to evict
                    needed = self.args.per_device_train_batch_size - len(slots)
                    extra_slots = self.active_streaming_batch.pick_random_slots(needed)
                    slots = torch.cat([slots, extra_slots]) if len(slots) > 0 else extra_slots

                if _do_debug:
                    print(f"[TRAINER DEBUG] Final selected slots={slots.tolist()}", flush=True)

                # Only run loss on selected slots (fixes OOM)
                loss, outputs = self.loss_fn(model, self.active_streaming_batch, slots=slots)

                if _do_debug:
                    print(
                        f"[TRAINER DEBUG] loss_fn returned loss={loss.item():.6f}  "
                        f"loss.requires_grad={loss.requires_grad}",
                        flush=True,
                    )
                
                # Meter update for processed slots
                with torch.no_grad():
                    batch = self.active_streaming_batch.get_batch(slots=slots)
                    m_logits = outputs.logits
                    m_labels = batch["labels"]
                    m_input_ids = batch["input_ids"]
                    m_maskable = m_labels != -100
                    m_masked = (m_input_ids == self.processing_class.mask_token_id) & m_maskable
                    m_nll = F.cross_entropy(m_logits.transpose(1, 2), m_labels, reduction="none", ignore_index=-100)
                    self.meter.update(
                        split="train" if model.training else "eval",
                        value=m_nll * m_masked.to(m_nll.dtype),
                        weight=m_maskable.to(dtype=m_logits.dtype)
                    )
                    # Update additional stats during evaluation
                    if not model.training and hasattr(outputs, "stats"):
                        for k, v in outputs.stats.items():
                            if v is not None:
                                self.meter.update_metric(split="eval", name=k, value=v)
            else:
                # Standard BPTT (step-wise unroll without persistence)
                inputs = self._preprocess_inputs(inputs)
                loss, outputs = self.loss_fn(model, inputs)
                # Meter update
                with torch.no_grad():
                    m_logits = outputs.logits
                    m_labels = inputs["labels"]
                    m_maskable = m_labels != -100
                    m_nll = F.cross_entropy(m_logits.transpose(1, 2), m_labels, reduction="none", ignore_index=-100)
                    self.meter.update(
                        split="train" if model.training else "eval",
                        value=m_nll,
                        weight=m_maskable.to(dtype=m_logits.dtype)
                    )
                    # Update additional stats during evaluation
                    if not model.training and hasattr(outputs, "stats"):
                        for k, v in outputs.stats.items():
                            if v is not None:
                                self.meter.update_metric(split="eval", name=k, value=v)

            # Note: Meters update for new losses might need specialized logic if needed,
            # but for now we follow the general trainer flow.
            return (loss, outputs) if return_outputs else loss
