import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class StreamingBatch:
    """
    Manages a buffer of partially masked sequences for Progressive UnMAsking (PUMA).
    """
    input_ids: torch.Tensor
    labels: torch.Tensor
    masked_mutable: torch.Tensor
    h_t: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_batch(cls, batch: Dict[str, torch.Tensor], mask_token_id: int):
        """
        Initializes a StreamingBatch from a standard batch.
        Assumes the input_ids are already 100% masked or masks them here.
        """
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"].clone()
        
        # In PUMA initialization, we start with 100% masks for maskable tokens
        maskable_mask = labels != -100
        input_ids[maskable_mask] = mask_token_id
        
        return cls(
            input_ids=input_ids,
            labels=labels,
            masked_mutable=maskable_mask,
            h_t=None,
            metadata={k: v for k, v in batch.items() if k not in ["input_ids", "labels"]}
        )

    def update(self, logits: torch.Tensor, threshold: float, h_s: Optional[torch.Tensor] = None):
        """
        Performs "On-Policy Unmasking" based on model confidence.
        """
        with torch.no_grad():
            # 1. Calculate uncertainty: 1 - max probability
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = probs.max(dim=-1)
            uncertainty = 1.0 - max_probs
            
            # 2. Sort by uncertainty (highest confidence first)
            # Only consider tokens that are currently masked and mutable
            uncertainty[~self.masked_mutable] = float("-inf")
            
            # We need to handle each sequence in the batch separately if they have different uncertainty profiles
            # For simplicity in this implementation, we apply the threshold per sequence
            for i in range(self.input_ids.shape[0]):
                u_i = uncertainty[i]
                mask_i = self.masked_mutable[i]
                
                if not mask_i.any():
                    continue
                
                # Sort mutable tokens by uncertainty
                vals, idxs = torch.sort(u_i[mask_i], descending=False)
                original_idxs = torch.where(mask_i)[0][idxs]
                
                # 3. Unmask all tokens whose cumulative uncertainty fits in the budget
                cum_uncertainty = torch.cumsum(vals, dim=-1)
                
                # Always unmask at least one token if something is still masked
                unmask_sub_idx = cum_uncertainty < threshold
                if not unmask_sub_idx.any():
                    unmask_sub_idx[0] = True
                
                to_unmask_idxs = original_idxs[unmask_sub_idx]
                
                # 4. Teacher Forcing: revealed tokens get ground-truth labels
                self.input_ids[i, to_unmask_idxs] = self.labels[i, to_unmask_idxs]
                self.masked_mutable[i, to_unmask_idxs] = False

        # 5. Update latent state
        if h_s is not None:
            self.h_t = h_s

    def is_finished(self) -> bool:
        """Returns True if all sequences in the batch are fully unmasked."""
        return not self.masked_mutable.any()

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Returns the current state as a standard batch dictionary."""
        batch = {
            "input_ids": self.input_ids,
            "labels": self.labels,
            **self.metadata
        }
        if self.h_t is not None:
            batch["h_t"] = self.h_t
        return batch
