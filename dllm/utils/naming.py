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

def flatten_config_dict(d, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = ["model_args", "gen_kwargs", "wandb_args"]
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in exclude_keys:
            if "active" in v:
                if k == "lora": flat["lora"] = v["active"]
                elif k == "cab": flat["use_cab"] = v["active"]
                elif k in ["puma", "bptt"]: flat[f"use_{k}"] = v["active"]
                elif k == "loophole": flat["use_loopholing"] = v["active"]
            for sub_k, sub_v in v.items():
                if sub_k != "active":
                    flat[sub_k] = sub_v
        else:
            flat[k] = v
    return flat

def get_experiment_naming(run_cfg, slurm_cfg):
    """
    Unified naming system for training runs.
    Returns: (group, run_name, tags, output_dir)
    """
    training = run_cfg.get("training", {})
    
    # 1. Base Model Identification
    model_path = str(training.get("model_name_or_path", "llada"))
    base_model = os.path.basename(model_path.rstrip("/"))

    # 1.5 Dataset Identification
    dataset_raw = str(training.get("dataset_args", "unknown"))
    dataset_slug = dataset_raw.split("[")[0].split("/")[-1]

    # 2. Extract Active Interventions
    use_lora = training.get("lora", False)
    use_loopholing = training.get("use_loopholing", False)
    use_cab = training.get("use_cab", False)
        
    loss_type = str(training.get("loss_type", "mlm")).lower()
    is_puma = "puma" in loss_type
    is_bptt = "bptt" in loss_type
    bptt_steps = _to_int(training.get("bptt_steps", 1))
    if bptt_steps > 1:
        is_bptt = True
        
    interventions_ordered = []
    if use_lora: interventions_ordered.append("lora")
    if use_loopholing: interventions_ordered.append("loophole")
    if is_puma: interventions_ordered.append("puma")
    if is_bptt: interventions_ordered.append("bptt")
    if use_cab: interventions_ordered.append("cab")
    
    if not use_lora:
        interv_str = "-".join(interventions_ordered) if interventions_ordered else ""
        training_mode = f"base-{interv_str}" if interv_str else "base"
    else:
        training_mode = "-".join(interventions_ordered)
        if not training_mode: training_mode = "base"
        
    group = f"{base_model}/{dataset_slug}/{training_mode}"

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
        
    if is_bptt:
        name_parts.append(f"bptt{bptt_steps}")
        
    if use_cab:
        b = training.get("cab_bottleneck_dim", 256)
        e = training.get("cab_mlp_expansion_dim", 512)
        rl = _to_int(training.get("read_layer", -1))
        cab_name = f"cab-b{b}-e{e}"
        if rl != -1:
            cab_name += f"-rl{rl}"
        name_parts.append(cab_name)
    elif use_loopholing:
        rl = _to_int(training.get("read_layer", -1))
        if rl != -1:
            name_parts.append(f"loop-rl{rl}")

    run_name = "_".join(name_parts)
    
    # 4. Final Metadata
    tags = sorted(list(set(interventions_ordered + [base_model, dataset_slug, "dllm"])))
    output_dir = f".models/{group}/{run_name}"
    return group, run_name, tags, output_dir

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
        
    # 2. Model and Checkpoint Slug Identification
    model_args = evaluation_cfg.get("model_args", {})
    pretrained = str(model_args.get("pretrained", "model")).strip("./")
    
    # Extract path nodes, removing common storage prefixes
    nodes = [n for n in pretrained.split("/") if n not in [".models", "models"]]
    
    checkpoint_name = "final"
    if nodes and nodes[-1].startswith("checkpoint-"):
        checkpoint_name = nodes.pop()
    
    # Construct model_slug using the remaining path parts (e.g. group/run_name)
    if len(nodes) >= 2:
        model_slug = os.path.join(*nodes[-2:]) # captures group and run_name
    elif nodes:
        model_slug = nodes[0]
    else:
        model_slug = "unknown_model"
    
    model_slug = model_slug.replace("/", "__") # Flatten for filesystem safety

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
    if temp > 0 or "temperature" in model_args: # Include t0.0 if explicitly set
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
    # Structure: <checkpoint_dir>/evals/<params_slug>/<task_slug>.jsonl
    # This avoids duplication nesting (task/task) since lm-eval-harness
    # appends the task name to the output_path directory.
    pretrained_raw = str(model_args.get("pretrained", "model"))
    base_eval_dir = os.path.join(pretrained_raw, "evals", params_slug)
    output_path = os.path.join(base_eval_dir, f"{task_slug}.jsonl")
    
    return task_slug, model_slug, checkpoint_name, params_slug, output_path
