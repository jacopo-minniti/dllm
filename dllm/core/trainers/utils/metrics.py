"""
Token-level NLL/PPL metrics for evaluation.

- NLLMetric: token-level mean negative log-likelihood (weighted mean over tokens).
- PPLMetric: exp(mean NLL) = perplexity.

Both use sync_on_compute=True so that compute() aggregates over all ranks.
"""

from typing import Union

import torch
import torchmetrics


class NLLMetric(torchmetrics.aggregation.MeanMetric):
    """Token-level mean NLL. Weights should be the mask of predicted (e.g. masked) tokens."""

    def __init__(self, **kwargs):
        # Ensure cross-rank aggregation when compute() is called
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)


class PPLMetric(NLLMetric):
    """Token-level perplexity = exp(mean NLL)."""

    def compute(self) -> torch.Tensor:
        mean_nll = super().compute()
        return torch.exp(mean_nll)


class StatsMetric(torchmetrics.aggregation.MeanMetric):
    """Simple mean metric for tracking scalar statistics."""

    def __init__(self, **kwargs):
        kwargs.setdefault("sync_on_compute", True)
        super().__init__(**kwargs)

    def update(self, value: Union[float, torch.Tensor]) -> None:
        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        super().update(value.to(self.device))
