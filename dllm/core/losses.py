import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class MLMLoss(nn.Module):
    def __init__(self, mask_token_id: int):
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, model, batch, **kwargs):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        
        maskable_mask = labels != -100
        # Standard MLM already happens in the trainer usually, 
        # but we provide it here for consistency if needed.
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
        h_t = None
        total_loss = 0
        
        # T-step unroll
        for t in range(self.num_steps):
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            
            # Loss at step t
            # For simplicity, we can use the same labels
            # In BPTT, we typically want to optimize the whole sequence
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask
            
            loss_t = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none"
            )
            loss_t = (loss_t * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
            total_loss += loss_t
            
            # Update input_ids for next step (Teacher Forcing a subset of tokens)
            # In a real BPTT loop for LLaDA, we might want to unmask some tokens
            # For now, let's assume a simple unmasking or just carry the latent
            if t < self.num_steps - 1:
                # Decide which tokens to unmask (e.g. random or confidence based)
                # Here we use a simple random unmasking of 50% of the remaining masks
                with torch.no_grad():
                    probs = torch.rand_like(masked_mask.float())
                    to_unmask = (probs < 0.5) & masked_mask
                    input_ids[to_unmask] = labels[to_unmask]
                
                h_t = h_s # Carry the latent state (keep the graph!)
                
        return total_loss / self.num_steps, outputs

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
        
        # In PUMA BPTT, we unroll num_steps on the same buffer
        for t in range(self.num_steps):
            batch = streaming_batch.get_batch(slots=slots)
            input_ids = batch["input_ids"].clone() # Clone to avoid in-place issues
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            h_t = batch.get("h_t")
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            final_outputs = outputs
            
            maskable_mask = labels != -100
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask
            
            loss_t = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=-100
            )

            if self.weighted_ce:
                # Per-position weight: w_i = (1 + conf_i)
                # Compute confidence using the selected metric
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

            # Fix average: Normalize by CURRENTLY masked tokens to match original repo
            loss_t = loss_t.sum() / masked_mask.float().sum().clamp_min(1)
            total_loss += loss_t
            
            # Update streaming batch with the new state (On-policy unmasking)
            # Within the BPTT loop, we keep gradients in h_s.
            # But the LAST step of the loop must detach if it is to be persisted for the NEXT global training call.
            # We pass detach=True for the last step.
            is_last_step = (t == self.num_steps - 1)
            h_s_to_store = h_s.detach() if is_last_step else h_s
            
            streaming_batch.update(
                logits, 
                threshold=self.threshold, 
                confidence_type=self.confidence_type,
                h_s=h_s_to_store,
                slots=slots
            )
            
            if streaming_batch.is_finished():
                break
                
        return total_loss / (t + 1), final_outputs
