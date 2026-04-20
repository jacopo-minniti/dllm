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

import datetime
import accelerate
from accelerate import PartialState
import transformers
import datasets
datasets.config.DEFAULT_NUM_PROC = 1

import dllm

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    use_chat_template: bool = field(
        default=True,
        metadata={"help": "Whether to use the chat template or raw text"},
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
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
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

    # Force single-threading to prevent fork/NCCL deadlocks
    data_args.num_proc = 1
    training_args.dataloader_num_workers = 0
    training_args.group_by_length = False 
    
    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    state = PartialState()
    def time_log(msg, force=False):
        if state.is_main_process or force:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Rank {state.process_index}: {msg}", flush=True)

    # 1. First sync to ensure all ranks are starting the data phase together
    time_log("Checkpoint: Starting dataset initialization...")
    state.wait_for_everyone() 
    
    # 2. All ranks load the raw dataset (each will wait for its own lock if needed)
    time_log("Checkpoint: Starting dataset initialization...")
    try:
        time_log(f"All ranks: Attempting to load dataset (HF_OFFLINE is now disabled)...")
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        time_log("All ranks: Raw dataset load succeeded.")
    except Exception as e:
        import traceback
        time_log(f"CRITICAL: Dataset loading failed on Rank {state.process_index}!")
        time_log(traceback.format_exc())
        time_log("CRASH ALERT: If you see 'ConnectionError', the local cache is incomplete and Rank 0 tried to reach the Hub.")
        raise e
    
    # 3. Handle Mapping/Processing
    import hashlib
    # Generate a unique hash for this dataset configuration to allow sharing across experiments
    # Added cache-version salt to force re-processing after prompt-masking fixes.
    cache_id = f"v4_{data_args.dataset}_{data_args.use_chat_template}_{data_args.mask_prompt_loss}_{model_args.model_name_or_path}"
    cache_hash = hashlib.md5(cache_id.encode()).hexdigest()
    processed_cache_path = os.path.join(".cache", "processed_datasets", cache_hash)
    
    if not data_args.load_preprocessed_data:
        # Check if already processed globally
        if os.path.exists(processed_cache_path) and os.path.exists(os.path.join(processed_cache_path, "dataset_info.json")):
            time_log(f"🚀 Found existing processed dataset in global cache: {processed_cache_path}")
            data_args.load_preprocessed_data = "SKIP_MAP" # Internal flag to skip mapping
        else:
            if state.is_main_process:
                time_log("Rank 0: Starting dataset mapping...")
                map_fn = partial(
                    dllm.utils.default_sft_map_fn,
                    tokenizer=tokenizer,
                    mask_prompt_loss=data_args.mask_prompt_loss,
                    use_chat_template=data_args.use_chat_template,
                )
                # Strict num_proc=1 after CUDA init
                dataset = dataset.map(
                    map_fn,
                    batched=False,
                    num_proc=1,
                    desc="Chat Template Mapping",
                )
                time_log("Rank 0: Mapping complete. Starting post_process_dataset...")
                dataset = dllm.utils.post_process_dataset(dataset, data_args)
                
                time_log(f"Rank 0: Saving processed dataset to shared cache at {processed_cache_path}...")
                os.makedirs(os.path.dirname(processed_cache_path), exist_ok=True)
                dataset.save_to_disk(processed_cache_path)
                time_log("Rank 0: Dataset saved to shared cache successfully.")
        
        # 4. Synchronize: all ranks wait for Rank 0 to finish mapping and saving (if it was needed)
        time_log("Waiting at the PRE-processing sync barrier...")
        state.wait_for_everyone()
        time_log("PRE-processing barrier cleared.")

        # 5. All ranks load the processed dataset from the shared cache.
        # Guard against a corrupted cache (e.g. the previous run was interrupted
        # while rank 0 was writing Arrow files). If loading fails, rank 0 removes
        # the corrupt directory, all ranks sync, then fall through to re-map.
        time_log(f"Rank {state.process_index}: Loading processed data from shared disk...")
        from datasets import load_from_disk
        import shutil
        _load_ok = False
        try:
            dataset = load_from_disk(processed_cache_path)
            _load_ok = True
            time_log(f"Rank {state.process_index}: Disk load complete.")
        except Exception as _load_err:
            time_log(f"WARNING: load_from_disk failed ({_load_err}). Treating cache as corrupt.", force=True)

        if not _load_ok:
            # Rank 0 removes the corrupt directory so the next barrier starts clean
            if state.is_main_process:
                if os.path.exists(processed_cache_path):
                    shutil.rmtree(processed_cache_path, ignore_errors=True)
                    time_log(f"Rank 0: Deleted corrupt cache at {processed_cache_path}.")
            state.wait_for_everyone()

            # Re-run mapping on rank 0 and save, then all ranks reload
            time_log("Rank 0: Re-mapping dataset after cache corruption...")
            if state.is_main_process:
                map_fn = partial(
                    dllm.utils.default_sft_map_fn,
                    tokenizer=tokenizer,
                    mask_prompt_loss=data_args.mask_prompt_loss,
                    use_chat_template=data_args.use_chat_template,
                )
                dataset = dllm.data.load_sft_dataset(
                    data_args.dataset,
                    load_preprocessed_data=False,
                )
                dataset = dataset.map(
                    map_fn,
                    batched=False,
                    num_proc=1,
                    desc="Chat Template Mapping (retry)",
                )
                dataset = dllm.utils.post_process_dataset(dataset, data_args)
                os.makedirs(os.path.dirname(processed_cache_path), exist_ok=True)
                dataset.save_to_disk(processed_cache_path)
                time_log("Rank 0: Re-mapping and save complete.")

            state.wait_for_everyone()
            time_log(f"Rank {state.process_index}: Loading re-processed data from shared disk...")
            dataset = load_from_disk(processed_cache_path)
            time_log(f"Rank {state.process_index}: Disk load complete (after cache recovery).")
    else:
        # If already preprocessed, we just ensure post-processing is consistent
        time_log("Using pre-processed flag. Finalizing dataset state...")
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # 6. Final sync before training starts
    time_log("Checkpoint: Entering final pre-training barrier...")
    state.wait_for_everyone()
    time_log("Checkpoint: All ranks synced. Releasing to trainer.")

    # ----- Training --------------------------------------------------------------
    logger.info("Start training...")
    # Fast dLLM utilizes the `151665` mask ID and handles noise masking internally inside its unified `forward` pass,
    # so we don't apply puma or mlm loss collators here by default, but rather the standard causal LM loss or custom 
    # internal loss (via `loss_type="puma"` etc). We keep the same Trainer pattern here.
    bd_size = getattr(model.config, "bd_size", 32)
    # Use max_length padding so the StreamingBatch (which fixes seq_len at init)
    # always sees consistent-length tensors across all batches.
    collate_fn = transformers.DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=data_args.max_length,
        pad_to_multiple_of=bd_size,
    )

    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=collate_fn,
    )
    try:
        # Handle cases where resume_from_checkpoint might be string "True"/"False" from CLI/YAML
        resume_checkpoint = training_args.resume_from_checkpoint
        if isinstance(resume_checkpoint, str):
            if resume_checkpoint.lower() == "true":
                resume_checkpoint = True
            elif resume_checkpoint.lower() == "false":
                resume_checkpoint = None

        if resume_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
        else:
            logger.info("Starting training from scratch.")

        trainer.train(resume_from_checkpoint=resume_checkpoint)
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
