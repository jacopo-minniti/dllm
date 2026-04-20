"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
import inspect
from dataclasses import dataclass
import torch.distributed as dist

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from .utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    threshold: float = 0.0  # PUMA threshold (0.0 means use fixed steps/scheduler)
    confidence_type: str = "top_prob"  # "top_prob", "prob_diff"


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        threshold = kwargs.get("threshold", config.threshold)
        confidence_type = kwargs.get("confidence_type", config.confidence_type)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"--- Sampling Configuration ---")
            print(f"Steps: {steps}, Max New Tokens: {max_new_tokens}, Block Size: {block_size}")
            print(f"Temperature: {temperature}, Remasking: {remasking}")
            print(f"CFG Scale: {cfg_scale}, Stochastic Transfer: {stochastic_transfer}")
            print(f"PUMA Threshold: {threshold}, Confidence Type: {confidence_type}")
            print(f"------------------------------")

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        # Also support EOT if available
        eot_id = getattr(self.tokenizer, "eot_token_id", None)

        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        # Ensure all inputs are tensors on the correct device
        if inputs:
            new_inputs = []
            for p in inputs:
                if not isinstance(p, torch.Tensor):
                    new_inputs.append(torch.as_tensor(p, dtype=torch.long, device=self.model.device))
                else:
                    new_inputs.append(p.to(self.model.device))
            inputs = new_inputs

        # Get prompt lengths
        prompt_lens = [p.shape[0] for p in inputs]
        
        # All ranks must agree on max prompt length even if some have 0 samples.
        if dist.is_initialized():
            local_max = max(prompt_lens) if prompt_lens else 0
            max_p_len_tensor = torch.tensor([local_max], device=self.model.device, dtype=torch.long)
            dist.all_reduce(max_p_len_tensor, op=dist.ReduceOp.MAX)
            max_prompt_len = int(max_p_len_tensor.item())
        else:
            max_prompt_len = max(prompt_lens) if prompt_lens else 0

        # Enforce that T is large enough for both the prompt and the new tokens
        if max_new_tokens:
            max_new_tokens = int(max_new_tokens)
            max_length = max_new_tokens + max_prompt_len
        elif max_length:
            max_length = int(max_length)
            # Safety: Ensure max_length doesn't truncate the prompt
            max_length = max(max_length, max_prompt_len + 1)
            max_new_tokens = max_length - max_prompt_len
        else:
            # Fallback
            max_new_tokens = 128
            max_length = max_prompt_len + max_new_tokens

        B = len(inputs)
        if dist.is_initialized() and B == 0:
            # Still participate in all_reduce even with empty batch to avoid deadlocks
            # The canvas T must be consistent with other ranks
            T = max_length
            return torch.zeros((0, T), dtype=torch.long, device=self.model.device)
        elif B == 0:
            return torch.zeros((0, 0), dtype=torch.long, device=self.model.device)

        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        # 🟢 Persistent mask tracking: use a boolean tensor to distinguish masks from EOS
        # especially when they share the same ID (e.g. in LLaDA-8B-Base).
        is_mask = torch.zeros((B, T), dtype=torch.bool, device=self.model.device)

        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            start, end = prompt_lens[i], prompt_lens[i] + max_new_tokens
            x[i, start:end] = mask_id  # append `max_new_tokens` masks to be generated
            is_mask[i, start:end] = True

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (~is_mask) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        is_mask[j, start:end]
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            if threshold <= 0:
                num_transfer_tokens = get_num_transfer_tokens(
                    mask_index=block_mask_index,
                    steps=steps,
                    scheduler=self.scheduler,
                    stochastic=stochastic_transfer,
                )
                # Some steps may be skipped if there are no transfers
                effective_steps = num_transfer_tokens.size(1)
            else:
                num_transfer_tokens = None
                effective_steps = steps

            # ----- Iterative reveal inside the current block -----
            # Initialize hidden states for loopholing memory across refinement steps
            h_t = None
            h_t_uncond = None

            # Robust check for h_t support in model signature (Outside the step loop for speed)
            forward_params = getattr(self, "_forward_params", None)
            if forward_params is None:
                # Resolve base model signature (handles PeftModel and other wrappers)
                model_to_inspect = self.model
                if hasattr(model_to_inspect, "get_base_model"): 
                    model_to_inspect = model_to_inspect.get_base_model()
                
                self._forward_params = inspect.signature(model_to_inspect.forward).parameters
                forward_params = self._forward_params
            
            can_use_h_t = "h_t" in forward_params
            use_loopholing = getattr(self.model.config, "use_loopholing", False)
            use_cab = getattr(self.model.config, "use_cab", False)
            # Either mechanism feeds h_s → h_t as cross-step memory
            use_h_t_memory = use_loopholing or use_cab

            for i in range(effective_steps):
                # Update mask_index within this block for this specific step
                mask_index = is_mask & attention_mask.bool()
                # within current block range
                block_start = max(0, min(prompt_lens) + b * block_size)
                # Actually we can just check if any masks are left in the whole sequence
                # but we focus on unmasking the currently targeted area.

                if not mask_index.any():
                    break

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Prepare h_t_batch for whichever mechanism is active.
                    # Must be defined BEFORE fwd_kwargs whether h_t is None or not
                    # (on step 0 h_t is None; on later steps it carries the cross-step memory).
                    if can_use_h_t:
                        if h_t is not None:
                            h_t_batch = torch.cat([h_t, h_t_uncond], dim=0)
                        else:
                            h_t_batch = None

                    # Forward pass
                    fwd_kwargs = {"input_ids": x_, "attention_mask": attention_mask.repeat(2, 1)}
                    if can_use_h_t:
                        fwd_kwargs["h_t"] = h_t_batch

                    m_out = self.model(**fwd_kwargs)
                    logits = m_out.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                    # Update h_t and h_t_uncond for next step (loopholing AND cab both use h_s)
                    if use_h_t_memory and can_use_h_t:
                        h_s = getattr(m_out, "h_s", None)
                        if h_s is not None:
                            h_t, h_t_uncond = torch.chunk(h_s, 2, dim=0)
                else:
                    # Forward pass
                    fwd_kwargs = {"input_ids": x, "attention_mask": attention_mask}
                    if can_use_h_t:
                        fwd_kwargs["h_t"] = h_t

                    m_out = self.model(**fwd_kwargs)
                    logits = m_out.logits

                    # Update h_t for next step (loopholing AND cab both use h_s)
                    if use_h_t_memory and can_use_h_t:
                        h_t = getattr(m_out, "h_s", None)

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Per-position confidence/uncertainty
                p = F.softmax(logits, dim=-1)
                if confidence_type == "top_prob":
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    uncertainty = 1.0 - x0_p
                elif confidence_type == "prob_diff":
                    top2_probs, _ = torch.topk(p, k=2, dim=-1)
                    conf = top2_probs[..., 0] - top2_probs[..., 1]
                    x0_p = conf
                    uncertainty = 1.0 - conf
                else:
                    raise ValueError(
                        f"Unsupported confidence_type={confidence_type!r}. "
                        "Supported values: top_prob, prob_diff."
                    )

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick which positions to commit
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                
                for j in range(B):
                    if threshold > 0:
                        # PUMA dynamic budgeting: cumulative uncertainty threshold
                        u_j = uncertainty[j][mask_index[j]]
                        if u_j.numel() == 0:
                            continue
                            
                        # Stable sort by uncertainty: add a tiny epsilon based on relative position
                        # to ensure deterministic tie-breaking even on GPU kernels.
                        u_j_stable = u_j + torch.linspace(0, 1e-8, u_j.numel(), device=u_j.device)
                        vals, sort_idx = torch.sort(u_j_stable, descending=False)
                        original_idxs = torch.where(mask_index[j])[0][sort_idx]
                        
                        cum_uncertainty = torch.cumsum(vals, dim=-1)
                        unmask_sub_idx = cum_uncertainty < threshold
                        
                        # Always unmask at least one token if we haven't reached the end
                        if not unmask_sub_idx.any():
                            unmask_sub_idx[0] = True
                            
                        to_unmask = original_idxs[unmask_sub_idx]
                        transfer_index[j, to_unmask] = True
                    else:
                        # Fixed budget scheduling (Standard LLaDA)
                        k = int(num_transfer_tokens[j, i].item())
                        if k > 0:
                            # Stable topk: add small epsilon to break ties deterministically
                            c_j = confidence[j]
                            c_j_stable = c_j + torch.linspace(0, 1e-8, c_j.numel(), device=c_j.device)
                            _, select_index = torch.topk(c_j_stable, k=k)
                            transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                is_mask[transfer_index] = False
                if histories is not None:
                    histories.append(x.clone())

                # Efficiency & EOS respect: If a sequence produces an EOS/EOT, fill trailing masks
                # for that sequence with EOS and check if the entire batch can stop early.
                # 🟢 Safety check: only optimize if mask and EOS are distinct.
                # If they are same, every mask token looks like an EOS, which triggers early exit.
                if mask_id != eos_id:
                    all_done = True
                    for j in range(B):
                        gen_part = x[j, prompt_lens[j]:]
                        # Check for termination tokens in the *revealed* part (not masks)
                        revealed_mask = ~is_mask[j, prompt_lens[j]:]
                        term_mask = (gen_part == eos_id) & revealed_mask
                        if eot_id is not None:
                            term_mask |= (gen_part == eot_id) & revealed_mask

                        term_indices = term_mask.nonzero(as_tuple=True)[0]
                        if term_indices.numel() > 0:
                            # 🟢 Ensure no masks remain before the EOS token before terminating
                            first_term_idx = prompt_lens[j] + term_indices[0].item()
                            if is_mask[j, prompt_lens[j]:first_term_idx].any():
                                all_done = False
                            else:
                                # Once we see a valid termination token, everything after it is irrelevant
                                if first_term_idx + 1 < T:
                                    x[j, first_term_idx + 1:] = eos_id
                                    is_mask[j, first_term_idx + 1:] = False
                        else:
                            all_done = False
                    
                    if all_done:
                        break
                else:
                    # If mask == EOS, we just check if no masks are left in the entire sequence.
                    if not is_mask.any():
                        break

            if all_done:
                break

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)
        # 🟢 Persistent mask tracking for infill
        is_mask = (torch.stack([
            F.pad(t, (0, T - t.shape[0]), value=eos_id) for t in inputs
        ]) == mask_id).to(self.model.device)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (~is_mask) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = is_mask[j, start : start + width]

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            # Initialize hidden states for loopholing memory across refinement steps
            h_t = None
            h_t_uncond = None

            # Robust check for h_t support in model signature (Outside the step loop for speed)
            forward_params = getattr(self, "_forward_params", None)
            if forward_params is None:
                # Resolve base model signature (handles PeftModel and other wrappers)
                model_to_inspect = self.model
                if hasattr(model_to_inspect, "get_base_model"): 
                    model_to_inspect = model_to_inspect.get_base_model()
                
                self._forward_params = inspect.signature(model_to_inspect.forward).parameters
            use_loopholing = getattr(self.model.config, "use_loopholing", False)
            can_use_h_t = "h_t" in forward_params

            for s in range(effective_steps):
                # Update mask_index within this block for this specific step
                mask_index_full = is_mask & attention_mask.bool()

                # Handle h_t carrying for CFG
                h_t_batch = None

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    if use_loopholing and h_t is not None and can_use_h_t:
                        h_t_batch = torch.cat([h_t, h_t_uncond], dim=0)

                    fwd_kwargs = {"input_ids": x_, "attention_mask": attention_mask.repeat(2, 1)}
                    if can_use_h_t:
                        fwd_kwargs["h_t"] = h_t_batch
                        
                    m_out = self.model(**fwd_kwargs)
                    logits = m_out.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                    # Update h_t and h_t_uncond
                    if use_loopholing and can_use_h_t:
                        h_s = getattr(m_out, "h_s", None)
                        if h_s is not None:
                            h_t, h_t_uncond = torch.chunk(h_s, 2, dim=0)
                else:
                    fwd_kwargs = {"input_ids": x, "attention_mask": attention_mask}
                    if can_use_h_t:
                        fwd_kwargs["h_t"] = h_t
                        
                    m_out = self.model(**fwd_kwargs)
                    logits = m_out.logits

                    # Update h_t
                    if use_loopholing and can_use_h_t:
                        h_t = getattr(m_out, "h_s", None)

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        # Stable topk for infill
                        c_j = confidence[j]
                        c_j_stable = c_j + torch.linspace(0, 1e-8, c_j.numel(), device=c_j.device)
                        _, select_idx = torch.topk(c_j_stable, k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                is_mask[transfer_index] = False
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)
