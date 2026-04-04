import hashlib
import os

def _to_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def _to_int(v, default=0):
    try:
        return int(v)
    except (ValueError, TypeError):
        return default

def get_experiment_naming(run_cfg, slurm_cfg):
    """
    Unified naming system for training runs.
    Returns: (group, run_name, tags, output_dir, run_id)
    """
    training = run_cfg.get("training", {})
    
    # 1. Base Model Identification
    model_path = str(training.get("model_name_or_path", "llada")).lower()
    if "llada" in model_path:
        if "instruct" in model_path:
            base_model = "llada-instruct"
        elif "base" in model_path:
            base_model = "llada-base"
        else:
            base_model = "llada"
    else:
        # Final fallback for local paths or other model types
        base_model = os.path.basename(model_path.rstrip("/")).replace("-", "").replace("_", "")

    # 1.5 Dataset Identification
    dataset_raw = str(training.get("dataset_args", "unknown"))
    if "tulu-3" in dataset_raw.lower():
        dataset_slug = "tulu3"
    elif "math500" in dataset_raw.lower():
        dataset_slug = "math500"
    else:
        # Fallback: extract last part of path but remove brackets/subsets and symbols
        dataset_slug = dataset_raw.split("[")[0].split("/")[-1].replace("-", "").replace("_", "").lower()

    # 2. Extract Active Interventions
    interventions = []
    
    use_lora = training.get("lora", False)
    if use_lora: interventions.append("lora")
    
    use_cab = training.get("use_cab", False)
    if use_cab: interventions.append("cab")
        
    loss_type = str(training.get("loss_type", "mlm")).lower()
    is_puma = "puma" in loss_type
    is_bptt = "bptt" in loss_type
    bptt_steps = _to_int(training.get("bptt_steps", 1))
    
    if is_puma: interventions.append("puma")
    if is_bptt: interventions.append("bptt")
    elif bptt_steps > 1: interventions.append("bptt")

    # Group: base_model-dataset-int1-int2 (sorted)
    group_parts = [base_model, dataset_slug]
    if interventions:
        group_parts.extend(sorted(interventions))
    else:
        group_parts.append("baseline")
        
    group = "-".join(group_parts)

    # 3. Construct Run Name (Hyperparams)
    name_parts = []
    
    # Essential Training Params
    lr = _to_float(training.get("learning_rate", 1e-5))
    lr_slug = f"{lr:g}"
    name_parts.append(f"lr{lr_slug}")
    
    # Calculate Effective Batch Size
    nodes = _to_int(slurm_cfg.get("nodes", 1))
    gpus_spec = str(slurm_cfg.get("gpus_per_node", "1"))
    try:
        gpus_per_node = int(gpus_spec.split(":")[-1]) if ":" in gpus_spec else int(gpus_spec)
    except:
        gpus_per_node = 1
        
    bs = _to_int(training.get("per_device_train_batch_size", 1))
    ga = _to_int(training.get("gradient_accumulation_steps", 1))
    eff_bs = nodes * gpus_per_node * bs * ga
    name_parts.append(f"bs{eff_bs}")

    # Intervention-Specific Params (Only if active)
    if use_lora:
        r = training.get("r", 32)
        alpha = training.get("lora_alpha", 64)
        name_parts.append(f"lora-r{r}-a{alpha}")
    
    if is_puma:
        th = training.get("puma_threshold", 0.15)
        name_parts.append(f"puma-th{th}")
        
    if "bptt" in interventions:
        name_parts.append(f"bptt{bptt_steps}")
        
    if use_cab:
        b = training.get("cab_bottleneck_dim", 256)
        e = training.get("cab_mlp_expansion_dim", 512)
        name_parts.append(f"cab-b{b}-e{e}")

    run_name = "_".join(name_parts)
    
    # 4. Final Metadata
    tags = sorted(list(set(interventions + [base_model, dataset_slug, "dllm"])))
    output_dir = f".models/{group}/{run_name}"
    run_id = hashlib.md5(f"{group}/{run_name}".encode()).hexdigest()

    return group, run_name, tags, output_dir, run_id

def get_eval_naming(evaluation_cfg):
    """
    Unified naming system for evaluation tasks.
    """
    # 1. Task Slug (e.g. math500_reasoning)
    tasks = evaluation_cfg.get("tasks", "eval")
    if isinstance(tasks, list):
        task_slug = "_".join(sorted(tasks))
    else:
        task_slug = str(tasks).replace(",", "_")
    task_slug = task_slug.replace("/", "__")
        
    # 2. Model and Checkpoint Slug
    model_args = evaluation_cfg.get("model_args", {})
    pretrained = str(model_args.get("pretrained", "model")).strip("./")
    
    # Normalize model path to group__name
    # Strip prefixes like .models, models, etc.
    nodes = [n for n in pretrained.split("/") if n not in [".models", "models"]]
    
    checkpoint_name = "final"
    if nodes and nodes[-1].startswith("checkpoint-"):
        checkpoint_name = nodes.pop()
    
    # The "model_slug" is strictly group__name
    if len(nodes) >= 2:
        model_slug = f"{nodes[-2]}__{nodes[-1]}"
    elif nodes:
        model_slug = nodes[0].replace("/", "__")
    else:
        model_slug = "default"

    # 3. Eval Params Slug
    eval_parts = []
    
    # Standard generation params
    mnt = _to_int(model_args.get("max_new_tokens", 0))
    if mnt > 0: eval_parts.append(f"mnt{mnt}")
    
    steps = _to_int(model_args.get("steps", 0))
    if steps > 0: eval_parts.append(f"s{steps}")
    
    bs = _to_int(model_args.get("block_size", 0))
    if bs > 0: eval_parts.append(f"bs{bs}")
    
    temp = _to_float(model_args.get("temperature", 0.0))
    eval_parts.append(f"t{temp}")
    
    # Diffusion/PUMA specific
    th = _to_float(model_args.get("threshold", 0.0))
    if th > 0: eval_parts.append(f"th{th}")
    
    if model_args.get("use_loopholing", False):
        eval_parts.append("loop")
        
    cfg = _to_float(model_args.get("cfg_scale", 0.0))
    if cfg > 0: eval_parts.append(f"cfg{cfg}")
    
    # Task specific
    nf = _to_int(evaluation_cfg.get("num_fewshot", 0))
    eval_parts.append(f"nf{nf}")
    
    params_slug = "_".join(eval_parts) if eval_parts else "default"
    
    # 4. Final Output Path
    # Structure: .evals/<task>/<group__name>/<checkpoint>/<params_slug>.jsonl
    base_eval_dir = os.path.join(".evals", task_slug, model_slug, checkpoint_name)
    output_path = os.path.join(base_eval_dir, f"{params_slug}.jsonl")
    
    return task_slug, model_slug, checkpoint_name, params_slug, output_path
