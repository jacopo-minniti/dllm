import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class MLMLoss(nn.Module):
    def __init__(self, mask_token_id: int):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, model, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        
        maskable_mask = labels != -100
        # Prime with masks if the input is clean (common in SFT collators)
        if not (input_ids == self.mask_token_id).any():
            input_ids[maskable_mask] = self.mask_token_id

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
            ignore_index=-100
        )
        
        # Only compute loss on masked tokens (that are maskable)
        masked_mask = (input_ids == self.mask_token_id) & maskable_mask
        loss = (loss * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
        
        return loss, outputs

class PumaLoss(nn.Module):
    def __init__(self, mask_token_id: int, threshold: float = 0.15, confidence_type: str = "top_prob"):
        super().__init__()
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
        logits = outputs.logits
        h_s = getattr(outputs, "h_s", None)
        
        # Loss calculation (on selected slots)
        maskable_mask = labels != -100
        masked_mask = (input_ids == self.mask_token_id) & maskable_mask
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none",
            ignore_index=-100
        )
        # Fix average: Normalize by CURRENTLY masked tokens to match original repo
        loss = loss.sum() / masked_mask.float().sum().clamp_min(1)
        
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
        # Prime with masks if input is clean
        if not (input_ids == self.mask_token_id).any():
            input_ids[maskable_mask] = self.mask_token_id

        h_t = None
        loss_terms = []
        
        # T-step unroll
        for t in range(self.num_steps):
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask
            if not masked_mask.any():
                break

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            
            loss_t = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=-100
            )
            # Gate loss by current masks
            loss_t = (loss_t * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
            loss_terms.append(loss_t)
            
            # Simple unmasking (50%) for the next step in the loop
            if t < self.num_steps - 1:
                with torch.no_grad():
                    probs = torch.rand_like(masked_mask.float())
                    to_unmask = (probs < 0.5) & masked_mask
                    input_ids[to_unmask] = labels[to_unmask]
                
                h_t = h_s 
                
        if not loss_terms:
            total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return total_loss, None

        return sum(loss_terms) / len(loss_terms), outputs

class LoopholingBPTTPumaLoss(nn.Module):
    def __init__(self, mask_token_id: int, threshold: float = 0.15, num_steps: int = 2, confidence_type: str = "top_prob", weighted_ce: bool = False):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.num_steps = num_steps
        self.confidence_type = confidence_type
        self.weighted_ce = weighted_ce

    def forward(self, model, streaming_batch, slots: Optional[torch.Tensor] = None, **kwargs):
        total_loss = 0
        final_outputs = None
        
    def forward(self, model, streaming_batch, slots: Optional[torch.Tensor] = None, **kwargs):
        batch = streaming_batch.get_batch(slots=slots)
        input_ids = batch["input_ids"].clone() 
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        h_t = batch.get("h_t")
        
        maskable_mask = labels != -100
        loss_terms = []
        final_outputs = None
        
        # In PUMA BPTT, we unroll num_steps on the same buffer
        for t in range(self.num_steps):
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask
            if not masked_mask.any():
                break

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            final_outputs = outputs
            
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
                    elif self.confidence_type == "entropy":
                        conf = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    else:
                        conf = probs.max(dim=-1)[0]
                    weights = 1.0 + conf
                loss_t = loss_t * weights

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
                        u = 1.0 - probs.max(dim=-1)[0]
                    
                    # For each sequence in the micro-batch, unmask some portion based on threshold
                    # This is a simplified version of the StreamingBatch.update logic
                    for b in range(input_ids.shape[0]):
                        m_i = masked_mask[b]
                        if not m_i.any(): continue
                        u_i = u[b][m_i]
                        v, idxs = torch.sort(u_i, descending=False)
                        target_indices = torch.where(m_i)[0][idxs]
                        # simplified cumsum thresholding
                        unmask_sub = torch.cumsum(v, dim=-1) < self.threshold
                        if not unmask_sub.any(): unmask_sub[0] = True
                        input_ids[b, target_indices[unmask_sub]] = labels[b, target_indices[unmask_sub]]

                h_t = h_s 
        
        if not loss_terms:
            total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return total_loss, final_outputs

        total_loss = sum(loss_terms) / len(loss_terms)

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
