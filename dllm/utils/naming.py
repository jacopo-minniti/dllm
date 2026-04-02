import hashlib
import os

def get_experiment_naming(run_cfg, slurm_cfg):
    """
    Unified naming system for training runs.
    Returns: (group, run_name, tags, output_dir, run_id)
    """
    training = run_cfg.get("training", {})
    
    # 1. Base Model Identification
    model_path = training.get("model_name_or_path", "llada")
    if "LLaDA" in model_path or "llada" in model_path.lower():
        base_model = "llada"
    else:
        base_model = os.path.basename(model_path.rstrip("/"))

    # 1.5 Dataset Identification
    dataset_raw = training.get("dataset_args", "unknown")
    if "tulu-3" in dataset_raw.lower():
        dataset_slug = "tulu3"
    elif "math500" in dataset_raw.lower():
        dataset_slug = "math500"
    else:
        # Fallback: extract last part of path but remove brackets/subsets
        dataset_slug = dataset_raw.split("[")[0].split("/")[-1].replace("-", "").lower()

    # 2. Extract Active Interventions
    interventions = []
    
    # LoRA
    use_lora = training.get("lora", False)
    if use_lora:
        interventions.append("lora")
    
    # CAB
    use_cab = training.get("use_cab", False)
    if use_cab:
        interventions.append("cab")
        
    # PUMA / BPTT
    loss_type = training.get("loss_type", "mlm")
    is_puma = "puma" in loss_type.lower()
    bptt_steps = int(training.get("bptt_steps", 1))
    
    if is_puma:
        interventions.append("puma")
    if bptt_steps > 1:
        interventions.append("bptt")

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
    lr = training.get("learning_rate", 1e-5)
    name_parts.append(f"lr{lr}")
    
    # Calculate Effective Batch Size
    nodes = int(slurm_cfg.get("nodes", 1))
    gpus_spec = str(slurm_cfg.get("gpus_per_node", "1"))
    try:
        gpus_per_node = int(gpus_spec.split(":")[-1]) if ":" in gpus_spec else int(gpus_spec)
    except:
        gpus_per_node = 1
        
    bs = int(training.get("per_device_train_batch_size", 1))
    ga = int(training.get("gradient_accumulation_steps", 1))
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
        
    if bptt_steps > 1:
        name_parts.append(f"bptt{bptt_steps}")
        
    if use_cab:
        b = training.get("cab_bottleneck_dim", 256)
        e = training.get("cab_mlp_expansion_dim", 512)
        name_parts.append(f"cab-b{b}-e{e}")

    run_name = "_".join(name_parts)
    
    # 4. Final Metadata
    tags = sorted(list(set(interventions + [base_model, "dllm"])))
    output_dir = f".models/{group}/{run_name}"
    run_id = hashlib.md5(f"{group}/{run_name}".encode()).hexdigest()

    return group, run_name, tags, output_dir, run_id

def get_eval_naming(evaluation_cfg, model_run_name_full=None):
    """
    Unified naming system for evaluation tasks.
    Returns: (task_slug, model_slug, checkpoint_name, params_slug, output_path)
    """
    # 1. Task Slug
    tasks = evaluation_cfg.get("tasks", "eval")
    if isinstance(tasks, list):
        task_slug = "_".join(sorted(tasks))
    else:
        task_slug = str(tasks).replace(",", "_")
        
    # 2. Model and Checkpoint Slug
    model_args = evaluation_cfg.get("model_args", {})
    pretrained = model_args.get("pretrained", "model")
    
    # Parse path like .models/llada-puma-cab/lr1e-5_bs128/checkpoint-1000
    path_parts = pretrained.strip("./").split("/")
    
    checkpoint_name = "final"
    if path_parts[-1].startswith("checkpoint-"):
        checkpoint_name = path_parts[-1]
        model_name_path = path_parts[:-1]
    else:
        model_name_path = path_parts

    # The "model_slug" is the group/name part
    if len(model_name_path) >= 2 and model_name_path[0] == ".models":
        model_slug = "__".join(model_name_path[1:])
    else:
        model_slug = "__".join(model_name_path).replace("/", "__")

    # 3. Eval Params Slug (Only relevant ones)
    eval_parts = []
    
    temp = float(model_args.get("temperature", 0.0))
    if temp > 0: eval_parts.append(f"t{temp}")
    
    steps = model_args.get("steps")
    if steps: eval_parts.append(f"s{steps}")
    
    th = float(model_args.get("threshold", 0.0))
    # If the model is a PUMA model, the threshold is relevant
    if th > 0 and ("puma" in model_slug.lower() or "puma" in checkpoint_name.lower()):
        eval_parts.append(f"th{th}")
        
    num_fewshot = evaluation_cfg.get("num_fewshot", 0)
    if num_fewshot > 0: eval_parts.append(f"nf{num_fewshot}")
    
    params_slug = "_".join(eval_parts) if eval_parts else "default"
    
    # 4. Final Output Path
    # .evals/<tasks>/<model_slug>/<checkpoint>/<params_slug>.jsonl
    base_eval_dir = os.path.join(".evals", task_slug, model_slug, checkpoint_name)
    output_path = os.path.join(base_eval_dir, f"{params_slug}.jsonl")
    
    return task_slug, model_slug, checkpoint_name, params_slug, output_path
