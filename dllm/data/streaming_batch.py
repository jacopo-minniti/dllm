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

    def initialize(self, batch: Dict[str, torch.Tensor], mask_token_id: int, capacity: Optional[int] = None):
        """
        Cold start: Initialize storage headers from the first batch and prime it.
        """
        if mask_token_id is None:
            raise ValueError(
                "StreamingBatch requires a valid `mask_token_id`, but found None. "
                "Ensure your tokenizer has a mask token configured (common for diffusion models like LLaDA or Fast-dLLM)."
            )
        batch_size, self.seq_len = batch["input_ids"].shape
        self.capacity = capacity if capacity is not None else batch_size
        device = batch["input_ids"].device
        
        # Initialize storage with capacity
        self.input_ids = torch.full((self.capacity, self.seq_len), mask_token_id, dtype=batch["input_ids"].dtype, device=device)
        self.labels = torch.full((self.capacity, self.seq_len), -100, dtype=batch["input_ids"].dtype, device=device)
        self.masked_mutable = torch.zeros((self.capacity, self.seq_len), dtype=torch.bool, device=device)
        self.h_t = None
        
        # Fill the first slots
        num_to_fill = min(batch_size, self.capacity)
        self.input_ids[:num_to_fill] = batch["input_ids"][:num_to_fill].clone()
        self.labels[:num_to_fill] = batch["labels"][:num_to_fill].clone()
        
        maskable_mask = self.labels[:num_to_fill] != -100
        self.input_ids[:num_to_fill][maskable_mask] = mask_token_id
        self.masked_mutable[:num_to_fill] = maskable_mask
        
        self.metadata = {}
        for k, v in batch.items():
            if k not in ["input_ids", "labels"] and isinstance(v, torch.Tensor):
                pad_val = 0 if "mask" in k.lower() else 0
                self.metadata[k] = torch.full((self.capacity, self.seq_len), pad_val, dtype=v.dtype, device=device)
                self.metadata[k][:num_to_fill] = v[:num_to_fill].clone()

        # ALL slots are ready to evict EXCEPT the ones we just filled
        self.ready_to_evict = torch.ones(self.capacity, dtype=torch.bool, device=device)
        self.ready_to_evict[:num_to_fill] = False

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


    def get_batch(self, slots: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Returns the buffer contents, optionally filtered to specific slots."""
        if slots is None:
            slots = slice(None)
            
        batch = {
            "input_ids": self.input_ids[slots],
            "labels": self.labels[slots],
            **{k: v[slots] for k, v in self.metadata.items()}
        }
        if self.h_t is not None:
            batch["h_t"] = self.h_t[slots]
        return batch

    def evict_and_fill(self, batch: Dict[str, torch.Tensor], mask_token_id: int, capacity: Optional[int] = None) -> torch.Tensor:
        """
        Replaces individual rows that have finished unmasking (ready_to_evict=True)
        with new ground-truth samples from the incoming batch.
        Returns the indices of the slots that were filled.
        """
        if self.input_ids is None:
            self.initialize(batch, mask_token_id, capacity=capacity)
            return torch.arange(min(batch["input_ids"].shape[0], self.capacity), device=batch["input_ids"].device)

        with torch.no_grad():
            # Find which slots in our buffer are empty/ready
            evict_idxs = torch.where(self.ready_to_evict)[0]
            num_to_fill = min(len(evict_idxs), batch["input_ids"].shape[0])
            
            if num_to_fill == 0:
                return torch.zeros(0, dtype=torch.long, device=batch["input_ids"].device)

            # Fill selected slots from the incoming batch
            target_slots = evict_idxs[:num_to_fill]
            
            # Prepare new tensors with correct sequence length
            new_input_ids = self._prepare_tensor(batch["input_ids"][:num_to_fill], self.seq_len, pad_value=0)
            new_labels = self._prepare_tensor(batch["labels"][:num_to_fill], self.seq_len, pad_value=-100)
            
            maskable_mask = new_labels != -100
            new_input_ids[maskable_mask] = mask_token_id
            
            self.input_ids[target_slots] = new_input_ids
            self.labels[target_slots] = new_labels
            self.masked_mutable[target_slots] = maskable_mask
            
            # Reset latent state for these slots with small noise to avoid RMSNorm zero-variance issues
            if self.h_t is not None:
                # Use same initialization scale as model: hidden_size**-0.5
                std = self.h_t.shape[-1] ** -0.5
                self.h_t[target_slots] = torch.randn_like(self.h_t[target_slots]) * std
            
            for k, v in batch.items():
                if k not in ["input_ids", "labels"] and k in self.metadata and isinstance(v, torch.Tensor):
                    pad_val = 0 if "mask" in k.lower() else 0
                    self.metadata[k][target_slots] = self._prepare_tensor(v[:num_to_fill], self.seq_len, pad_value=pad_val)

            # These slots are no longer ready to evict (they just started)
            self.ready_to_evict[target_slots] = False
            
            return target_slots

    def pick_random_slots(self, count: int) -> torch.Tensor:
        """Pick random rows that are NOT yet ready to evict."""
        not_ready = torch.where(~self.ready_to_evict)[0]
        if len(not_ready) == 0:
            return torch.arange(min(count, self.capacity), device=self.ready_to_evict.device)
        perm = torch.randperm(len(not_ready), device=not_ready.device)
        return not_ready[perm[:count]]

    def update(self, logits: torch.Tensor, threshold: float, confidence_type: str = "top_prob", slots: Optional[torch.Tensor] = None, h_s: Optional[torch.Tensor] = None):
        """
        Performs "On-Policy Unmasking" based on model confidence and updates eviction state.
        Only processes the specified slots.
        """
        if slots is None:
            slots = torch.arange(self.capacity, device=logits.device)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            
            if confidence_type == "top_prob":
                max_probs, _ = probs.max(dim=-1)
                uncertainty = 1.0 - max_probs
            elif confidence_type == "prob_diff":
                top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                conf = top2_probs[..., 0] - top2_probs[..., 1]
                uncertainty = 1.0 - conf
            else:
                raise ValueError(
                    f"Unsupported confidence_type={confidence_type!r}. "
                    "Supported values: top_prob, prob_diff."
                )
            
            for k, i in enumerate(slots.tolist()):
                u_i = uncertainty[k]
                mask_i = self.masked_mutable[i]
                
                if not mask_i.any():
                    self.ready_to_evict[i] = True
                    continue
                
                vals, idxs = torch.sort(u_i[mask_i], descending=False)
                original_idxs = torch.where(mask_i)[0][idxs]
                
                cum_uncertainty = torch.cumsum(vals, dim=-1)
                unmask_sub_idx = cum_uncertainty < threshold
                if not unmask_sub_idx.any():
                    unmask_sub_idx[0] = True
                
                to_unmask_idxs = original_idxs[unmask_sub_idx]
                self.input_ids[i, to_unmask_idxs] = self.labels[i, to_unmask_idxs]
                self.masked_mutable[i, to_unmask_idxs] = False
                
                if not self.masked_mutable[i].any():
                    self.ready_to_evict[i] = True

        if h_s is not None:
            if torch.isnan(h_s).any():
                nan_rows = torch.isnan(h_s).flatten(1).any(dim=1)
                h_s = h_s.clone()
                h_s[nan_rows] = 0.0
            if self.h_t is None:
                b, l, d = h_s.shape
                self.h_t = torch.zeros((self.capacity, self.seq_len, d), dtype=h_s.dtype, device=h_s.device)
            self.h_t[slots] = h_s

    def is_finished(self) -> bool:
        """Returns True if the entire buffer is ready to evict."""
        return self.ready_to_evict.all() if self.ready_to_evict is not None else True
