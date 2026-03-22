import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

class StreamingBatch:
    """
    Manages a persistent buffer of partially masked sequences for Progressive UnMAsking (PUMA).
    Supports row-wise eviction and filling for maximum throughput efficiency.
    """
    def __init__(self):
        self.input_ids: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.masked_mutable: Optional[torch.Tensor] = None
        self.h_t: Optional[torch.Tensor] = None
        self.metadata: Dict[str, Any] = {}
        
        # Track which rows are finished and can be replaced by new data
        self.ready_to_evict: Optional[torch.Tensor] = None
        self.capacity: int = 0
        self.seq_len: int = 0

    def initialize(self, batch: Dict[str, torch.Tensor], mask_token_id: int):
        """
        Cold start: Initialize storage headers from the first batch and prime it.
        """
        self.capacity, self.seq_len = batch["input_ids"].shape
        device = batch["input_ids"].device
        
        self.input_ids = batch["input_ids"].clone()
        self.labels = batch["labels"].clone()
        
        # In PUMA initialization, we start with 100% masks for maskable tokens
        maskable_mask = self.labels != -100
        self.input_ids[maskable_mask] = mask_token_id
        
        self.masked_mutable = maskable_mask
        self.h_t = None
        self.metadata = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items() if k not in ["input_ids", "labels"]}
        
        # Initially, nothing is ready to evict (we just started)
        self.ready_to_evict = torch.zeros(self.capacity, dtype=torch.bool, device=device)

    def _prepare_tensor(self, t: torch.Tensor, target_len: int, pad_value: Any) -> torch.Tensor:
        """Pad or truncate a tensor to the target sequence length."""
        curr_len = t.shape[-1]
        if curr_len == target_len:
            return t
        if curr_len < target_len:
            # Pad on the right (last dimension)
            return torch.nn.functional.pad(t, (0, target_len - curr_len), value=pad_value)
        else:
            # Truncate
            return t[..., :target_len]

    def evict_and_fill(self, batch: Dict[str, torch.Tensor], mask_token_id: int):
        """
        Replaces individual rows that have finished unmasking (ready_to_evict=True)
        with new ground-truth samples from the incoming batch.
        """
        if self.input_ids is None:
            self.initialize(batch, mask_token_id)
            return self.capacity

        with torch.no_grad():
            # Find which slots in our buffer are empty/ready
            evict_idxs = torch.where(self.ready_to_evict)[0]
            num_to_fill = min(len(evict_idxs), batch["input_ids"].shape[0])
            
            if num_to_fill == 0:
                return 0

            # Fill selected slots from the incoming batch
            target_slots = evict_idxs[:num_to_fill]
            
            # Prepare new tensors with correct sequence length
            # Note: input_ids will be masked below, so pad value here is temporary
            new_input_ids = self._prepare_tensor(batch["input_ids"][:num_to_fill], self.seq_len, pad_value=0)
            new_labels = self._prepare_tensor(batch["labels"][:num_to_fill], self.seq_len, pad_value=-100)
            
            maskable_mask = new_labels != -100
            new_input_ids[maskable_mask] = mask_token_id
            
            self.input_ids[target_slots] = new_input_ids
            self.labels[target_slots] = new_labels
            self.masked_mutable[target_slots] = maskable_mask
            
            # Reset latent state for these slots
            if self.h_t is not None:
                self.h_t[target_slots] = 0.0
            
            for k, v in batch.items():
                if k not in ["input_ids", "labels"] and k in self.metadata and isinstance(v, torch.Tensor):
                    # Guess logical pad value: 0 for attention_mask, etc.
                    pad_val = 0 if "mask" in k.lower() else 0
                    self.metadata[k][target_slots] = self._prepare_tensor(v[:num_to_fill], self.seq_len, pad_value=pad_val)

            # These slots are no longer ready to evict (they just started)
            self.ready_to_evict[target_slots] = False
            
            return num_to_fill

    def update(self, logits: torch.Tensor, threshold: float, h_s: Optional[torch.Tensor] = None):
        """
        Performs "On-Policy Unmasking" based on model confidence and updates eviction state.
        """
        with torch.no_grad():
            # 1. Calculate uncertainty: 1 - max probability
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = probs.max(dim=-1)
            uncertainty = 1.0 - max_probs
            
            # 2. Sort by uncertainty (highest confidence first)
            uncertainty[~self.masked_mutable] = float("-inf")
            
            for i in range(self.capacity):
                u_i = uncertainty[i]
                mask_i = self.masked_mutable[i]
                
                if not mask_i.any():
                    self.ready_to_evict[i] = True
                    continue
                
                # Sort mutable tokens by uncertainty
                vals, idxs = torch.sort(u_i[mask_i], descending=False)
                original_idxs = torch.where(mask_i)[0][idxs]
                
                # 3. Unmask all tokens whose cumulative uncertainty fits in the budget
                cum_uncertainty = torch.cumsum(vals, dim=-1)
                
                unmask_sub_idx = cum_uncertainty < threshold
                if not unmask_sub_idx.any():
                    unmask_sub_idx[0] = True
                
                to_unmask_idxs = original_idxs[unmask_sub_idx]
                
                # 4. Teacher Forcing
                self.input_ids[i, to_unmask_idxs] = self.labels[i, to_unmask_idxs]
                self.masked_mutable[i, to_unmask_idxs] = False
                
                # 5. Check if this row is now finished
                if not self.masked_mutable[i].any():
                    self.ready_to_evict[i] = True

        # 6. Update latent state
        if h_s is not None:
            self.h_t = h_s

    def is_finished(self) -> bool:
        """Returns True if the entire buffer is ready to evict (rarely used in row-wise mode)."""
        return self.ready_to_evict.all() if self.ready_to_evict is not None else True

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
