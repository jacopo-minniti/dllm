from typing import Dict, Iterable, Mapping, Optional, Union
import copy

import torch
import transformers
import torchmetrics


class BaseMetricsCallback(transformers.TrainerCallback):
    """
    Split-aware torchmetrics callback.

    Provide metrics via:
      - metrics=MetricCollection(...)
      - metrics={"name": Metric(), ...}

    Metrics are cloned (or deep-copied) per split, moved to accelerator.device,
    and reset on init.

    Finalize:
      - compute+reset on *all* ranks
      - log/print only on main process
    """

    def __init__(
        self,
        trainer: "transformers.Trainer",
        splits: Iterable[str] = ("train", "eval"),
        metrics: torchmetrics.MetricCollection | dict[str, torchmetrics.Metric] = None,
    ):
        super().__init__()
        self.trainer = trainer
        self.accelerator = trainer.accelerator
        self.splits = tuple(splits)

        if isinstance(metrics, dict):
            metrics = torchmetrics.MetricCollection(dict(metrics))

        assert isinstance(metrics, torchmetrics.MetricCollection)

        self._m: dict[str, torchmetrics.MetricCollection] = {}
        device = self.accelerator.device

        for s in self.splits:
            self._m[s] = copy.deepcopy(metrics)
            self._m[s].to(device)
            self._m[s].reset()

    @staticmethod
    def key_for(split: str, name: str) -> str:
        return name if split == "train" else f"{split}_{name}"

    @torch.no_grad()
    def update(self, split: str, *args, **kwargs) -> None:
        self._m[split].update(*args, **kwargs)

    @torch.no_grad()
    def finalize(self, split: str) -> dict[str, float]:
        """Compute metrics and reset. Must be called on all ranks so sync aggregates over all ranks."""
        mc = self._m[split]
        mc.to(self.accelerator.device)

        # Metrics with sync_on_compute=True (e.g. NLLMetric/PPLMetric) sync state across
        # ranks in compute(), so the returned value is the same on every rank.
        computed = mc.compute()
        mc.reset()

        return {
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in computed.items()
        }

    @torch.no_grad()
    def log_and_print(
        self,
        state: transformers.TrainerState,
        splits: Iterable[str] | None = None,
    ) -> None:
        splits = self.splits if splits is None else tuple(splits)

        # All ranks finalize to make metric sync/compute safe.
        vals = {s: self.finalize(s) for s in splits}

        if not self.accelerator.is_main_process:
            return

        logs: dict[str, float] = {}
        for s, d in vals.items():
            for k, v in d.items():
                logs[self.key_for(s, k)] = v

        if logs:
            self.trainer.log(logs)
            print(
                f"[step {state.global_step} epoch {state.epoch}] "
                + " ".join(f"{k}={v:.6f}" for k, v in logs.items())
            )



class OnEvaluateMetricsCallback(BaseMetricsCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.log_and_print(state, splits=("train", "eval"))
        return control


class WandbAlertCallback(transformers.TrainerCallback):
    """
    Sends an alert message when the run starts, provided that WandB is used.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if (
            state.is_world_process_zero
            and args.report_to
            and "wandb" in args.report_to
        ):
            try:
                import wandb

                run = wandb.run
                if run is not None:
                    wandb.alert(
                        title=f"Run Started: {run.name}",
                        text=(
                            f"Training has begun on {run.host}.\n"
                            f"View here: {run.get_url()}"
                        ),
                        level=wandb.AlertLevel.INFO,
                        wait_duration=0,  # Send immediately
                    )
            except (ImportError, Exception):
                pass
        return control
