import sys
import yaml

from dllm.utils.config import load_resolved_config, resolve_keywords
from dllm.utils.naming import flatten_config_dict

with open("configs/keywords.yaml", "r") as f:
    keyword_map = yaml.safe_load(f)

base_run_cfg = load_resolved_config("configs/train/base.yaml", "configs/train", "../default.yaml")
if "training" in base_run_cfg:
    base_run_cfg["training"] = flatten_config_dict(base_run_cfg["training"])

# emulate the extra_args
# suppose the user passed NO extra args.
# resolve keywords
res = resolve_keywords(base_run_cfg, keyword_map)

import json
print(json.dumps(res, indent=2))
