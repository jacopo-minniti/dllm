"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --load_in_4bit True --lora True

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/sft.py

Slurm users
# Note: run `mkdir .logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"
"""

import os
import sys
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = ".models/LLaDA-8B-Base/tulu-3-sft-mixture[train:10000,test:1000]"
    group_by_length: bool = True
    num_train_epochs: float = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Handle SIGTERM (e.g. Slurm pre-emption or time limit)
    import signal
    import wandb
    import time
    def handle_sigterm(signum, frame):
        if accelerate.PartialState().is_main_process:
            if wandb.run is not None:
                group = os.getenv('WANDB_RUN_GROUP', 'none')
                wandb.alert(
                    title="⚠️ Run Terminated (SIGTERM)",
                    text=f"*Group*: `{group}`\n*Status*: `Pre-empted or Timeout`",
                    level=wandb.AlertLevel.WARN
                )
                time.sleep(2)
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    # Use global rank 0 first to download and prepare the dataset (especially for multi-node)
    state = accelerate.PartialState()
    with state.main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            # Only use multiple processes for mapping on the main process to avoid thundering herd on cache locks
            # For non-main processes, they should just load the cached version.
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc if state.is_main_process else 1,
                desc="Mapping dataset to SFT format",
            )
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(  # padded <eos_token> should be visible
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,  # finetune on padded <eos_token>
                ),
            )
        ),
    )
    try:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
        trainer.processing_class.save_pretrained(
            os.path.join(training_args.output_dir, "checkpoint-final")
        )
    except Exception as e:
        if accelerate.PartialState().is_main_process:
            import wandb
            import time
            if wandb.run is not None:
                group = os.getenv('WANDB_RUN_GROUP', 'none')
                wandb.alert(
                    title="❌ Run Failed",
                    text=f"*Group*: `{group}`\n*Error*: `{str(e)[:500]}`",
                    level=wandb.AlertLevel.ERROR
                )
                time.sleep(5)  # Give time for the alert to be sent
        raise e


if __name__ == "__main__":
    train()
