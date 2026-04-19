import os
from dataclasses import dataclass, field

import transformers

from .utils import get_default_logger, resolve_with_base_env

logger = get_default_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = None  # overwrite this
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str = None
    # --- fold PEFT args here ---
    lora: bool = False
    target_modules: str = "all-linear"
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    bias: str = "none"
    modules_to_save: str = None
    eval_checkpoint: str = None
    merge_lora: bool = False
    # Architectural flags — default False.  models.py converts False → None so that a
    # flag absent from the config defers to the model's own saved config.json, while
    # True always overrides.  (There is no way to force-disable via config; just don't
    # load a checkpoint that has the flag enabled.)
    use_loopholing: bool = False
    only_mask_tokens: bool = False
    mlp_module: bool = False
    use_cab: bool = False
    use_puma: bool = False
    use_bptt: bool = False
    cab_bottleneck_dim: int = 128
    cab_mlp_expansion_dim: int = 512
    read_layer: list[int] = field(default_factory=lambda: [-1])
    read_layers: list[int] = field(default=None)
    freeze_backbone: bool = True
    cab_n_heads: int = 8
    cab_n_kv_heads: int = 4
    # --- model-level dropout ---
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1

    def __post_init__(self):
        # Sync read_layer and read_layers
        if self.read_layers is not None:
             self.read_layer = self.read_layers
        
        self.model_name_or_path = resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )
        # Force backbone dropouts to 0 when backbone is frozen — dropout on frozen weights wastes nothing
        # but can hurt gradient flow through the active module. Applies to all frozen-backbone modes.
        has_module = self.use_cab or self.use_loopholing
        if has_module and not self.lora and self.freeze_backbone:
            self.attention_dropout = 0.0
            self.residual_dropout = 0.0
            self.embedding_dropout = 0.0


@dataclass
class DataArguments:
    dataset: str = None  # overwrite this
    num_proc: int = 8
    disable_caching: bool = False
    max_length: int = 1024
    truncation: str = field(
        default="right",
        metadata={
            "help": (
                'The truncation strategy to use ("filter" or "right"). '
                '"filter" only keeps sequences that are shorter than max_length; '
                '"right" only keeps the rightmost max_length tokens for each sequence.'
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = None  # overwrite this
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    num_train_epochs: float = 10
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_steps: float = 10
    eval_on_start: bool = False
    eval_strategy: str = "steps"
    eval_steps: float = 0.1
    save_steps: float = 0.1
    save_only_model: bool = False
    resume_from_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to a previous checkpoint or 'True' to find latest."},
    )
    ignore_data_skip: bool = field(
        default=False, 
        metadata={"help": "Avoid CPU RAM spikes during resume by not fast-forwarding data."}
    )
    ddp_timeout: int = field(
        default=1800,
        metadata={"help": "The timeout for `torch.distributed.init_process_group` in seconds."}
    )

    def __post_init__(self):
        super().__post_init__()
        # Priority: explicit CLI arg > WANDB_NAME env var > output_dir
        self.run_name = self.run_name or os.getenv("WANDB_NAME") or self.output_dir
        if self.group_by_length:
            logger.info(
                "training_args.group_by_length=True: preprocessing "
                "may take some time after `trainer.train()` starts."
            )
