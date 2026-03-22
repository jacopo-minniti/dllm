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

from ..schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.configs import TrainingArguments
from dllm.utils.data import prepend_bos
from .utils import NLLMetric, PPLMetric, OnEvaluateMetricsCallback, WandbAlertCallback
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


class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        args: MDLMConfig,
        scheduler: BaseAlphaScheduler | None = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)

        if not (0.0 < args.time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        self.time_epsilon = args.time_epsilon
        self.loss_weight_type = args.loss_weight_type
        self.loss_norm_type = args.loss_norm_type
        self.right_shift_logits = args.right_shift_logits

        self.meter = OnEvaluateMetricsCallback(
            trainer=self,
            splits=("train", "eval"),
            metrics={"nll": NLLMetric(), "ppl": PPLMetric()},
        )
        self.add_callback(self.meter)
        self.add_callback(WandbAlertCallback())

        # Registry of loss functions
        self.loss_fns = {
            "mlm": MLMLoss(self.processing_class.mask_token_id),
            "puma": PumaLoss(self.processing_class.mask_token_id, threshold=args.puma_threshold),
            "bptt": LoopholingBPTTLoss(self.processing_class.mask_token_id, num_steps=args.bptt_steps),
            "puma_bptt": LoopholingBPTTPumaLoss(self.processing_class.mask_token_id, threshold=args.puma_threshold, num_steps=args.bptt_steps),
        }
        self.loss_fn = self.loss_fns.get(args.loss_type, self.loss_fns["mlm"])
        self.active_streaming_batch = None

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
                self.active_streaming_batch.evict_and_fill(
                    inputs, 
                    self.processing_class.mask_token_id,
                    capacity=capacity
                )
                
                loss, outputs = self.loss_fn(model, self.active_streaming_batch)
                # Meter update
                with torch.no_grad():
                    batch = self.active_streaming_batch.get_batch()
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

            # Note: Meters update for new losses might need specialized logic if needed,
            # but for now we follow the general trainer flow.
            return (loss, outputs) if return_outputs else loss
