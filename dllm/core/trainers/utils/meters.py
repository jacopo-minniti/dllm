from typing import Dict, Iterable, Mapping, Optional, Union
import copy
import os

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
    Sends alerts to WandB (and thus Slack if configured) for run events.
    """

    def _send_alert(self, title: str, text: str, level: str = "info"):
        try:
            import wandb
            # Check if run exists and is active. Sometimes on_train_begin is too early.
            run = wandb.run
            if run is not None:
                # Map string level to wandb.AlertLevel
                alert_level = {
                    "info": wandb.AlertLevel.INFO,
                    "warn": wandb.AlertLevel.WARN,
                    "error": wandb.AlertLevel.ERROR,
                }.get(level.lower(), wandb.AlertLevel.INFO)

                wandb.alert(
                    title=title,
                    text=text,
                    level=alert_level,
                    wait_duration=0,
                )
                import time
                time.sleep(2)  # Give time to send
        except (ImportError, Exception):
            pass

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and args.report_to and "wandb" in args.report_to:
            group = os.getenv('WANDB_RUN_GROUP', 'none')
            tags = os.getenv('WANDB_TAGS', 'none')
            self._send_alert(
                title="🚀 Run Started",
                text=f"*Group*: `{group}`\n*Tags*: `{tags}`\n*Output Dir*: `{args.output_dir}`",               
                level="info"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and args.report_to and "wandb" in args.report_to:
            group = os.getenv('WANDB_RUN_GROUP', 'none')
            self._send_alert(
                title="✅ Run Success",
                text=f"*Group*: `{group}`\n*Steps*: `{state.global_step}`\n*Status*: `Completed`",
                level="info"
            )
        return control

    def on_train_error(self, args, state, control, **kwargs):
        if state.is_world_process_zero and args.report_to and "wandb" in args.report_to:
            group = os.getenv('WANDB_RUN_GROUP', 'none')
            self._send_alert(
                title="❌ Run Failed",
                text=f"*Group*: `{group}`\n*Last Step*: `{state.global_step}`\n*Status*: `Crashed`",
                level="error"
            )
        return control


class SlurmCheckpointCallback(transformers.TrainerCallback):
    """
    Callback that catches SIGTERM (sent by Slurm when a job is about to time out)
    and triggers a checkpoint save before exiting gracefully.
    """

    def __init__(self):
        self.sigterm_received = False
        import signal
        import sys
        
        # Register the signal handler on all ranks
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        self.sigterm_received = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.sigterm_received:
            # Tell the trainer to save a full checkpoint and stop training
            # Use 'SIGTERM' as the marker in the logs
            print(f"Rank {args.local_rank} received SIGTERM (Slurm timeout). Requesting checkpoint save at step {state.global_step}...")
            control.should_save = True
            control.should_training_stop = True
            
            # Optionally send a WandB alert if we are on the main process
            if state.is_world_process_zero:
                try:
                    import wandb
                    if wandb.run is not None:
                        group = os.getenv('WANDB_RUN_GROUP', 'none')
                        wandb.alert(
                            title="⚠️ Slurm Job Timeout/Preemption",
                            text=f"*Group*: `{group}`\n*Step*: `{state.global_step}`\n*Status*: `Saving checkpoint and exiting...`",
                            level=wandb.AlertLevel.WARN
                        )
                except:
                    pass
        return control
