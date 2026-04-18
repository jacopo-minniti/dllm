import yaml
import json
from dllm.utils.config import load_resolved_config, resolve_keywords
from dllm.utils.naming import flatten_config_dict

# Mock keyword_map
with open("configs/keywords.yaml", "r") as f:
    keyword_map = yaml.safe_load(f)

# Load cab_fast.yaml
base_run_cfg = load_resolved_config("configs/train/cab_fast.yaml", "configs/train", "../default.yaml")

if "training" in base_run_cfg:
    base_run_cfg["training"] = flatten_config_dict(base_run_cfg["training"])

# Resolve keywords
res_cfg = resolve_keywords(base_run_cfg, keyword_map)

# Print relevant values
training = res_cfg.get("training", {})
print(f"Model Path: {training.get('model_name_or_path')}")
print(f"Script Path: {training.get('script_path')}")
print(f"Learning Rate: {training.get('learning_rate')}")
print(f"Epochs: {training.get('num_train_epochs')}")
