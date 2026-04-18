import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


SUPPORTED_CONFIDENCE_TYPES = {"top_prob", "prob_diff"}


def _validate_confidence_type(confidence_type: str) -> None:
    if confidence_type not in SUPPORTED_CONFIDENCE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_CONFIDENCE_TYPES))
        raise ValueError(
            f"Unsupported confidence_type={confidence_type!r}. "
            f"Supported values: {supported}."
        )


class MLMLoss(nn.Module):
    def __init__(self, mask_token_id: int):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, model, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        
        maskable_mask = labels != -100
        # Always mask target positions in non-streaming losses to ensure a valid signal
        input_ids[maskable_mask] = self.mask_token_id

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
            ignore_index=-100
        )
        
        # Original MLM: Global sequence mean (including zero loss at unmasked/ignored positions)
        loss = (loss * maskable_mask.float()).sum() / input_ids.numel()
        
        return loss, outputs

class PumaLoss(nn.Module):
    def __init__(self, mask_token_id: int, threshold: float = 0.15, confidence_type: str = "top_prob"):
        super().__init__()
        _validate_confidence_type(confidence_type)
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.confidence_type = confidence_type

    def forward(self, model, streaming_batch, slots: Optional[torch.Tensor] = None, **kwargs):
        batch = streaming_batch.get_batch(slots=slots)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        h_t = batch.get("h_t")
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t, labels=None)
        logits = outputs.logits
        h_s = getattr(outputs, "h_s", None)
        
        # Attach stats for logging
        outputs.stats = {
            "intervention_ratio": getattr(outputs, "intervention_ratio", None),
            "gamma_mean": getattr(outputs, "gamma_mean", None),
            "h_s_norm": h_s.norm(p=2, dim=-1).mean().item() if h_s is not None else None,
            "h_t_norm": h_t.norm(p=2, dim=-1).mean().item() if h_t is not None else None,
        }
        
        # Loss calculation (on selected slots)
        maskable_mask = labels != -100
        masked_mask = (input_ids == self.mask_token_id) & maskable_mask
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
            ignore_index=-100
        )
        # Original PumaLoss: Per-token average over currently masked tokens
        loss = (loss * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
        
        # Update streaming batch (On-policy unmasking) - only for these slots
        streaming_batch.update(
            logits, 
            slots=slots, 
            threshold=self.threshold, 
            confidence_type=self.confidence_type,
            h_s=h_s.detach() if h_s is not None else None
        )
        
        return loss, outputs

class LoopholingBPTTLoss(nn.Module):
    def __init__(self, mask_token_id: int, num_steps: int = 2):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps

    def forward(self, model, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        
        maskable_mask = labels != -100
        # Always mask target positions in non-streaming BPTT to ensure a valid signal
        input_ids[maskable_mask] = self.mask_token_id

        # Collect stats
        stats = {
            "intervention_ratio": [],
            "gamma_mean": [],
            "h_s_norm": [],
            "h_t_norm": [],
        }

        h_t = None
        loss_terms = []
        
        # T-step unroll
        for t in range(self.num_steps):
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t, labels=None)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            
            # Capture stats
            if getattr(outputs, "intervention_ratio", None) is not None:
                stats["intervention_ratio"].append(outputs.intervention_ratio)
            if getattr(outputs, "gamma_mean", None) is not None:
                stats["gamma_mean"].append(outputs.gamma_mean)
            if h_s is not None:
                stats["h_s_norm"].append(h_s.norm(p=2, dim=-1).mean().item())
            if h_t is not None:
                stats["h_t_norm"].append(h_t.norm(p=2, dim=-1).mean().item())

            if not masked_mask.any():
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"BPTT step {t}: zero masked tokens (mask_token_id={self.mask_token_id}). "
                    "Skipping loss for this step. If this persists, check dataset labels."
                )
                dummy_loss = 0.0 * logits.sum()
                loss_terms.append(dummy_loss)
                break
            
            loss_t = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=-100
            )
            # Original BPTT: Per-token average over currently masked tokens in this step
            loss_t = (loss_t * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
            loss_terms.append(loss_t)
            
            # Simple unmasking (50%) for the next step in the loop
            if t < self.num_steps - 1:
                with torch.no_grad():
                    probs = torch.rand_like(masked_mask.float())
                    to_unmask = (probs < 0.5) & masked_mask
                    # Use out-of-place update to avoid autograd version mismatch
                    input_ids = torch.where(to_unmask, labels, input_ids)
                
                h_t = h_s 
                
        # Original BPTT: Sum of step-wise averages
        total_loss = sum(loss_terms)
        
        # Attach averaged stats to final outputs
        outputs.stats = {k: (sum(v)/len(v) if v else None) for k, v in stats.items()}
        return total_loss, outputs

class LoopholingBPTTPumaLoss(nn.Module):
    def __init__(self, mask_token_id: int, threshold: float = 0.15, num_steps: int = 2, confidence_type: str = "top_prob", weighted_ce: bool = False):
        super().__init__()
        _validate_confidence_type(confidence_type)
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.num_steps = num_steps
        self.confidence_type = confidence_type
        self.weighted_ce = weighted_ce

    def forward(self, model, streaming_batch, slots: Optional[torch.Tensor] = None, **kwargs):
        batch = streaming_batch.get_batch(slots=slots)
        input_ids = batch["input_ids"].clone() 
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        h_t = batch.get("h_t")
        
        maskable_mask = labels != -100
        # Collect stats
        stats = {
            "intervention_ratio": [],
            "gamma_mean": [],
            "h_s_norm": [],
            "h_t_norm": [],
        }

        loss_terms = []

        # In PUMA BPTT, we unroll num_steps on the same buffer
        for t in range(self.num_steps):
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t, labels=None)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            final_outputs = outputs
            
            # DEBUG
            if t == 0:
                print(f"[DEBUG Loss] Step {t}: masked_mask.sum={masked_mask.sum().item()}, maskable_mask.sum={maskable_mask.sum().item()}, input_ids.masks={(input_ids == self.mask_token_id).sum().item()}")
                print(f"[DEBUG Loss] Step {t}: logits mean={logits.mean().item()}, std={logits.std().item()}")
                # Check if labels are all -100
                print(f"[DEBUG Loss] Step {t}: labels != -100 count={(labels != -100).sum().item()}")

            # Capture stats
            if getattr(outputs, "intervention_ratio", None) is not None:
                stats["intervention_ratio"].append(outputs.intervention_ratio)
            if getattr(outputs, "gamma_mean", None) is not None:
                stats["gamma_mean"].append(outputs.gamma_mean)
            if h_s is not None:
                stats["h_s_norm"].append(h_s.norm(p=2, dim=-1).mean().item())
            if h_t is not None:
                stats["h_t_norm"].append(h_t.norm(p=2, dim=-1).mean().item())

            if not masked_mask.any():
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    f"PUMA-BPTT step {t}: zero masked tokens "
                    f"(mask_token_id={self.mask_token_id}, "
                    f"input_ids masks={( input_ids == self.mask_token_id).sum().item()}, "
                    f"maskable={maskable_mask.sum().item()}). "
                    "Skipping loss for this step. If this persists, check dataset labels."
                )
                dummy_loss = 0.0 * logits.sum()
                loss_terms.append(dummy_loss)
                break
            
            loss_t = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=-100
            )

            if self.weighted_ce:
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    if self.confidence_type == "top_prob":
                        conf = probs.max(dim=-1)[0]
                    elif self.confidence_type == "prob_diff":
                        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                        conf = top2_probs[..., 0] - top2_probs[..., 1]
                    else:
                        raise ValueError(
                            f"Unsupported confidence_type={self.confidence_type!r}"
                        )
                    weights = 1.0 + conf
                loss_t = loss_t * weights

            # Original Puma BPTT: Per-token average over currently masked tokens in this step
            loss_t = (loss_t * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
            loss_terms.append(loss_t)
            
            # Update local state for next step in the BPTT unroll
            if t < self.num_steps - 1:
                # Use same confidence-based sampling as PUMA to decide local unmasking
                with torch.no_grad():
                    # We compute unmasking on-policy
                    probs = torch.softmax(logits, dim=-1)
                    if self.confidence_type == "top_prob":
                        u = 1.0 - probs.max(dim=-1)[0]
                    elif self.confidence_type == "prob_diff":
                        top2, _ = torch.topk(probs, k=2, dim=-1)
                        u = 1.0 - (top2[..., 0] - top2[..., 1])
                    else:
                        raise ValueError(
                            f"Unsupported confidence_type={self.confidence_type!r}"
                        )
                    
                    # For each sequence in the micro-batch, unmask some portion based on threshold
                    # Use out-of-place update to avoid autograd version mismatch (indices are saved by Embedding)
                    new_input_ids = input_ids.clone()
                    for b in range(input_ids.shape[0]):
                        m_i = masked_mask[b]
                        if not m_i.any(): continue
                        u_i = u[b][m_i]
                        v, idxs = torch.sort(u_i, descending=False)
                        target_indices = torch.where(m_i)[0][idxs]
                        # simplified cumsum thresholding
                        unmask_sub = torch.cumsum(v, dim=-1) < self.threshold
                        if not unmask_sub.any(): unmask_sub[0] = True
                        new_input_ids[b, target_indices[unmask_sub]] = labels[b, target_indices[unmask_sub]]
                    input_ids = new_input_ids

                h_t = h_s 
        
        # Original Puma BPTT: Sum of step-wise averages
        total_loss = sum(loss_terms)
        
        # Attach averaged stats to final outputs
        final_outputs.stats = {k: (sum(v)/len(v) if v else None) for k, v in stats.items()}

        # Update the persistent buffer with the final state from the loop
        with torch.no_grad():
            streaming_batch.update(
                logits, 
                threshold=self.threshold, 
                confidence_type=self.confidence_type,
                h_s=h_s.detach() if h_s is not None else None,
                slots=slots
            )
            
        return total_loss, final_outputs
