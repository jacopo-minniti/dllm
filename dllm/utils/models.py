import os
from types import SimpleNamespace

import accelerate
import torch
import transformers
from peft import prepare_model_for_kbit_training

from .configs import ModelArguments, TrainingArguments
from .utils import disable_caching_allocator_warmup, load_peft, print_main


def get_model(
    model_args: ModelArguments | None = None,
    config: transformers.PretrainedConfig | None = None,
    **kwargs,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible input sources.

    Args:
        model_args: Dataclass or namespace containing model parameters, or None to use **kwargs.
        config: Optional transformers.PretrainedConfig to use instead of loading from the checkpoint.
        **kwargs: Override or supply params when model_args is None (e.g. model_name_or_path, dtype).

    Returns:
        transformers.PreTrainedModel
    """
    model_args = model_args or ModelArguments()
    model_name_or_path = kwargs.get(
        "model_name_or_path", getattr(model_args, "model_name_or_path", None)
    )
    dtype = kwargs.get("dtype", getattr(model_args, "dtype", "bfloat16"))
    load_in_4bit = kwargs.get(
        "load_in_4bit", getattr(model_args, "load_in_4bit", False)
    )
    attn_implementation = kwargs.get(
        "attn_implementation", getattr(model_args, "attn_implementation", None)
    )

    # Device map: skip when ZeRO-3 or FSDP is enabled to allow proper sharding
    ps = accelerate.PartialState()
    is_sharded = (
        transformers.integrations.is_deepspeed_zero3_enabled() or 
        ps.distributed_type == accelerate.utils.DistributedType.FSDP
    )
    
    device_map = (
        {"": ps.local_process_index}
        if not is_sharded and torch.cuda.is_available()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Map dtype string to torch.dtype
    torch_dtype = dtype
    if isinstance(dtype, str):
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32

    params = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    # Ensure local paths are recognized as such by transformers (starting with ./ or absolute)
    if model_name_or_path and not model_name_or_path.startswith("/"):
        if model_name_or_path.startswith(".") or os.path.isdir(model_name_or_path):
             model_name_or_path = os.path.abspath(model_name_or_path)

    # Detect if we are loading a PEFT checkpoint
    is_peft = False
    if model_name_or_path and os.path.isdir(model_name_or_path):
        if os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
            is_peft = True
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_path = peft_config.base_model_name_or_path
            # If base model is also a local path, make it absolute
            if base_model_path and not base_model_path.startswith("/"):
                if base_model_path.startswith(".") or os.path.isdir(base_model_path):
                    base_model_path = os.path.abspath(base_model_path)
            print_main(f"ℹ️ Detected PEFT checkpoint. Loading base model: {base_model_path}")
        else:
            base_model_path = model_name_or_path
    else:
        base_model_path = model_name_or_path

    # Final absolute path guarantee before calling transformers
    if base_model_path and not base_model_path.startswith("/"):
        if base_model_path.startswith(".") or os.path.isdir(base_model_path):
            base_model_path = os.path.abspath(base_model_path)

    # Pre-fetch remote models on rank 0 to avoid race conditions and host RAM OOM
    if base_model_path and not os.path.isdir(base_model_path) and not base_model_path.startswith(("/", ".")):
        ps = accelerate.PartialState()
        if ps.num_processes > 1:
            if ps.is_main_process:
                print_main(f"ℹ️ Rank 0 is pre-fetching remote model: {base_model_path}")
                # Trigger download of config at least
                transformers.AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
            ps.wait_for_everyone()

    # --- Checkpoint Self-Containment Check ---
    # If base_model_path is a directory, ensure it has the LLaDA code files for trust_remote_code
    if os.path.isdir(base_model_path) and not os.path.exists(os.path.join(base_model_path, "configuration_llada.py")):
        # Only do this for LLaDA models
        is_llada = False
        if os.path.exists(os.path.join(base_model_path, "config.json")):
            with open(os.path.join(base_model_path, "config.json"), "r") as f:
                import json
                if json.load(f).get("model_type") == "llada": 
                    is_llada = True
        
        if is_llada:
            model_src = "dllm/pipelines/llada/models/"
            if os.path.exists(model_src):
                print_main(f"ℹ️ Rescuing missing modeling files in {base_model_path}...")
                import shutil
                for f in os.listdir(model_src):
                    if f.endswith(".py"):
                        shutil.copy2(os.path.join(model_src, f), base_model_path)

    try:
        # Detect model type to force local implementation if it's LLaDA
        # this ensures local features like loopholing (h_t) are available
        check_config = config or transformers.AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        
        # Propagate all custom settings to the config object strictly
        # This avoids passing them as kwargs to from_pretrained which can cause __init__ errors
        custom_fields = {
            "use_loopholing": getattr(model_args, "use_loopholing", False),
            "only_mask_tokens": getattr(model_args, "only_mask_tokens", False),
            "mlp_module": getattr(model_args, "mlp_module", False),
            "use_cab": getattr(model_args, "use_cab", False),
            "cab_bottleneck_dim": getattr(model_args, "cab_bottleneck_dim", 128),
            "cab_mlp_expansion_dim": getattr(model_args, "cab_mlp_expansion_dim", 512),
            "read_layers": getattr(model_args, "read_layer", [-1]),
            "cab_n_heads": getattr(model_args, "cab_n_heads", 8),
            "cab_n_kv_heads": getattr(model_args, "cab_n_kv_heads", 4),
            "attention_dropout": getattr(model_args, "attention_dropout", None),
            "residual_dropout": getattr(model_args, "residual_dropout", None),
            "embedding_dropout": getattr(model_args, "embedding_dropout", None),
        }
        for field_name, value in custom_fields.items():
            if value is not None:
                setattr(check_config, field_name, value)
                
        if getattr(check_config, "model_type", None) == "llada":
             from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
             print_main(f"ℹ️ Forcing local LLaDAModelLM. Config: CAB={check_config.use_cab}, Loop={check_config.use_loopholing}")
             model = LLaDAModelLM.from_pretrained(base_model_path, config=check_config, **params)
        else:
             model = transformers.AutoModelForMaskedLM.from_pretrained(
                 base_model_path, config=check_config, **params
             )
    except Exception:
        model = transformers.AutoModel.from_pretrained(base_model_path, **params)

    if is_peft:
        print_main(f"ℹ️ Loading PEFT adapter from: {model_name_or_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_name_or_path)
        # Sync after heavy PEFT loading
        ps = accelerate.PartialState()
        if ps.num_processes > 1:
            ps.wait_for_everyone()
        print_main("✅ PEFT adapter loaded successfully.")

    # --- if quantized, prepare for LoRA / QLoRA training ---
    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Optionally train with lora: Only if we haven't already loaded a PEFT model
    if not is_peft:
        model = load_peft(model, model_args)
    else:
        print_main("ℹ️ Skipping load_peft because model was already loaded from PEFT checkpoint.")

    if getattr(model_args, "lora", False) and not load_in_4bit:
        # Avoid the error: element 0 of tensors does not require grad and does not have a grad_fn
        # This is needed because input_ids (integers) do not require gradients, and
        # torch's gradient checkpointing (use_reentrant=True) requires at least one input with requires_grad=True
        # to record the graph of operations in the checkpointed block.
        # This is already handled by prepare_model_for_kbit_training, but not for regular LoRA.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            # fallback
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Optionally merge LoRA weights for inference speedups
    if is_peft and getattr(model_args, "merge_lora", False):
        print_main("ℹ️ Merging LoRA weights for faster inference...")
        try:
            model = model.merge_and_unload()
            print_main("✅ LoRA weights merged successfully.")
        except Exception as e:
            print_main(f"⚠️ Failed to merge LoRA weights: {e}")

    # --- Freeze backbone and only train CAB if requested ---
    if getattr(model_args, "use_cab", False) and not getattr(model_args, "lora", False):
        print_main("❄️ CAB training detected. Freezing base model parameters...")
        for name, param in model.named_parameters():
            if "cab" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print_main(f"🔥 Training parameter: {name}")

        # ensure input require grads for checkpointing support
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


def get_tokenizer(
    model_args: ModelArguments | None = None, **kwargs
) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with flexible input sources.

    Args:
        model_args: Namespace/dataclass containing at least model_name_or_path, or None to use **kwargs.
        **kwargs: Override or supply params when model_args is None (e.g. model_name_or_path).

    Returns:
        transformers.PreTrainedTokenizer
    """
    # Lazy imports to avoid circular dependencies
    from transformers import (
        BertPreTrainedModel,
        ModernBertPreTrainedModel,
        RobertaPreTrainedModel,
    )

    from dllm.pipelines.a2d import (
        A2DLlamaLMHeadModel,
        A2DQwen2LMHeadModel,
        A2DQwen3LMHeadModel,
    )
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM

    model_args = model_args or ModelArguments()
    model_name_or_path = kwargs.get(
        "model_name_or_path", getattr(model_args, "model_name_or_path", None)
    )

    # Ensure local path is treated as such by transformers
    if model_name_or_path and not model_name_or_path.startswith("/"):
        if model_name_or_path.startswith(".") or os.path.isdir(model_name_or_path):
            model_name_or_path = os.path.abspath(model_name_or_path)

    # ---------------- Tokenizer loading ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )

    assert tokenizer.eos_token is not None or tokenizer.pad_token is not None

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    # If model is not provided, return as-is
    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    try:
        model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]
    except (KeyError, AttributeError):
        # Fallback for local models not registered in AutoModel mapping
        model_cls = None
        if getattr(model_cfg, "model_type", None) == "llada":
            from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
            model_cls = LLaDAModelLM

    # Identify model family to apply specific tokenizer customizations
    model_type = getattr(model_cfg, "model_type", None)

    # ---------------- Model-specific customization ----------------
    is_llada = (model_type == "llada" or (model_cls and issubclass(model_cls, LLaDAModelLM)))
    is_llada_moe = (model_type in ["llada_moe", "llada2_moe"] or (model_cls and issubclass(model_cls, (LLaDAMoEModelLM, LLaDA2MoeModelLM))))
    is_dream = (model_cls and issubclass(model_cls, DreamModel))
    is_bert_family = (model_cls and issubclass(model_cls, (BertPreTrainedModel, RobertaPreTrainedModel, ModernBertPreTrainedModel)))
    is_a2d_llama = (model_cls and issubclass(model_cls, A2DLlamaLMHeadModel))
    is_a2d_qwen = (model_cls and issubclass(model_cls, (A2DQwen2LMHeadModel, A2DQwen3LMHeadModel)))

    if is_llada:
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        
        tokenizer.eot_token = "<|eot_id|>"
        # tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token) # can not do this for llada base directly
        # TODO: for llada base, add special_tokens = {"<|start_header_id|>": 126346, "<|end_header_id|>": 126347, "<|eot_id|>": 126348}
        # fix bugs in chat template
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    elif is_llada_moe:
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif is_dream:
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif is_bert_family:
        tokenizer.eot_token = "[/Answer]"
        tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    elif is_a2d_llama:
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif is_a2d_qwen:
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
        # When enable_thinking is not passed, default to False so the chat template
        # appends <think></think> (add think). Only skip that when enable_thinking=True.
        _orig_apply_chat_template = tokenizer.apply_chat_template

        def _apply_chat_template(*args, **kwargs):
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
            try:
                return _orig_apply_chat_template(*args, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return _orig_apply_chat_template(*args, **kwargs)

        tokenizer.apply_chat_template = _apply_chat_template
    else:
        print_main("no tokenizer customization for model class:", model_cls)
    return tokenizer
