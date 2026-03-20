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
            input_ids,
            reduction="none"
        )
        
        # Only compute loss on masked tokens (that are maskable)
        masked_mask = (input_ids == self.mask_token_id) & maskable_mask
        loss = (loss * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
        
        return loss, outputs

class PumaLoss(nn.Module):
    def __init__(self, mask_token_id: int, threshold: float = 0.15):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.threshold = threshold

    def forward(self, model, streaming_batch, **kwargs):
        batch = streaming_batch.get_batch()
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        h_t = batch.get("h_t")
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
        logits = outputs.logits
        h_s = getattr(outputs, "h_s", None)
        
        # Loss calculation
        maskable_mask = labels != -100
        masked_mask = (input_ids == self.mask_token_id) & maskable_mask
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            reduction="none"
        )
        loss = (loss * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
        
        # Update streaming batch (On-policy unmasking)
        streaming_batch.update(logits, threshold=self.threshold, h_s=h_s)
        
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
    def __init__(self, mask_token_id: int, threshold: float = 0.15, num_steps: int = 2):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.threshold = threshold
        self.num_steps = num_steps

    def forward(self, model, streaming_batch, **kwargs):
        total_loss = 0
        final_outputs = None
        
        for t in range(self.num_steps):
            batch = streaming_batch.get_batch()
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            h_t = batch.get("h_t")
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, h_t=h_t)
            logits = outputs.logits
            h_s = getattr(outputs, "h_s", None)
            final_outputs = outputs
            
            # Loss at step t
            maskable_mask = labels != -100
            masked_mask = (input_ids == self.mask_token_id) & maskable_mask
            
            loss_t = F.cross_entropy(
                logits.transpose(1, 2),
                labels,
                reduction="none"
            )
            loss_t = (loss_t * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1)
            total_loss += loss_t
            
            # Update streaming batch (On-policy unmasking)
            streaming_batch.update(logits, threshold=self.threshold, h_s=h_s)
            
            # If the batch is finished, we can break early
            if streaming_batch.is_finished():
                break
                
        return total_loss / (t + 1), final_outputs
