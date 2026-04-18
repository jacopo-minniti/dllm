import os
import yaml
import copy
from typing import Any, Dict, List, Generator

def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges source dict into target dict."""
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            deep_merge(target[k], v)
        else:
            target[k] = copy.deepcopy(v)
    return target

def load_resolved_config(path: str, base_dir: str, default_filename: str = "default.yaml") -> Dict[str, Any]:
    """
    Loads a config, merges it with the base default.yaml if it exists.
    """
    # 1. Load Defaults
    default_path = os.path.join(base_dir, default_filename)
    merged_cfg = {}
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            merged_cfg = yaml.safe_load(f) or {}
    
    # 2. Load User Config
    # Check if path exists as-is (e.g. if user passed the full path)
    actual_path = path
    if not os.path.exists(actual_path):
        # Try relative to base_dir
        actual_path = os.path.join(base_dir, path)
    
    # Try adding .yaml extension
    if not os.path.exists(actual_path) and not actual_path.endswith(".yaml"):
        if os.path.exists(actual_path + ".yaml"):
            actual_path += ".yaml"
    
    if os.path.exists(actual_path):
        with open(actual_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        merged_cfg = deep_merge(merged_cfg, user_cfg)
    else:
        # If still not found, we don't error but it will just be the defaults
        print(f"⚠️ Warning: Configuration file not found at {path} or {actual_path}")
    
    return merged_cfg

def resolve_keywords(config: Any, keyword_map: Dict[str, Any]) -> Any:
    """
    Recursively replaces keyword strings and expands shortcuts into multiple parameters.
    """
    if isinstance(config, dict):
        new_dict = {}
        # Track keys set by shortcuts so they aren't overwritten by original keys
        expanded_keys = set()
        
        # We use a copy of the items to avoid modification issues
        items = list(config.items())
        idx = 0
        while idx < len(items):
            k, v = items[idx]
            
            # 1. Handle Categorical Shortcuts (e.g., cab_size: medium)
            shortcuts = keyword_map.get("shortcuts", {})
            if k in shortcuts:
                if isinstance(v, list) and len(v) > 0:
                    # Multi-value shortcut (Matrix)
                    # Resolve one element to identify the target parameters
                    first_res = resolve_keywords({k: v[0]}, keyword_map)
                    for target_k in first_res.keys():
                        new_dict[target_k] = [resolve_keywords({k: item}, keyword_map).get(target_k) for item in v]
                        expanded_keys.add(target_k)
                    idx += 1
                    continue
                
                # Single-value shortcut
                if not isinstance(v, (dict, list)):
                    try:
                        if v in shortcuts[k]:
                            expansion = shortcuts[k][v]
                            if isinstance(expansion, dict):
                                for exp_k, exp_v in expansion.items():
                                    new_dict[exp_k] = resolve_keywords(exp_v, keyword_map)
                                    expanded_keys.add(exp_k)
                            else:
                                new_dict[k] = expansion
                                expanded_keys.add(k)
                            idx += 1
                            continue
                    except TypeError:
                        pass

            # 2. Handle Layers (base 1 to base 0 conversion)
            if k in ["read_layers", "read_layer"] and v is not None:
                if k not in expanded_keys:
                    if isinstance(v, list):
                        new_dict[k] = [(x - 1 if isinstance(x, int) else x) for x in v]
                    elif isinstance(v, int):
                        new_dict[k] = v - 1
                    else:
                        new_dict[k] = v
            
            # 3. Handle standard list-based matrix shorthand (nested resolution)
            elif isinstance(v, list):
                if k not in expanded_keys:
                    new_dict[k] = [resolve_keywords(x, keyword_map) for x in v]
            
            # 4. Standard recursive resolution
            else:
                if k not in expanded_keys:
                    new_dict[k] = resolve_keywords(v, keyword_map)
            
            idx += 1
        return new_dict

    elif isinstance(config, list):
        return [resolve_keywords(v, keyword_map) for v in config]

    elif isinstance(config, str):
        # Check for direct matches in keywords.yaml datasets/models
        if config in keyword_map.get("datasets", {}):
            return keyword_map["datasets"][config]
        if config in keyword_map.get("models", {}):
            return keyword_map["models"][config]
        
    return config

def expand_matrix_config(config: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Finds lists in the config and explodes them into multiple 'concrete' configs.
    Uses Zipped iteration (parallel lists must have same length).
    """
    # 1. Flatten to find all list values and their paths
    flat_lists = []
    
    def find_lists(obj, path=None):
        path = path or []
        if isinstance(obj, dict):
            for k, v in obj.items():
                find_lists(v, path + [k])
        elif isinstance(obj, list) and len(obj) > 1:
            # We treat a list as a matrix dimension if it has more than 1 element
            # UNLESS it is specifically nested, e.g. [[1, 2]] means a single job with [1,2].
            
            # Special case: If it's a list of lists where every sub-list has 1 element, OR if it's a list of numbers.
            # To sweep over multiple layers together, use: read_layers: [[15, 16], [30, 31]]
            # To use multiple layers in a single job without expansion, use: read_layers: [[15, 16]]
            
            # If the first element is a list, and the whole list has length 1, it's a 'locked' value.
            if len(obj) == 1 and isinstance(obj[0], list):
                return
                
            flat_lists.append({"path": path, "values": obj})

    find_lists(config)
    
    if not flat_lists:
        yield config
        return
    
    # 2. Validate lengths
    lengths = [len(x["values"]) for x in flat_lists]
    max_len = max(lengths)
    for x in flat_lists:
        if len(x["values"]) != max_len and len(x["values"]) != 1:
            raise ValueError(
                f"Matrix dimension mismatch at {'.'.join(x['path'])}. "
                f"Expected length {max_len} or 1, got {len(x['values'])}"
            )
    
    # 3. Yield exploded configs
    for i in range(max_len):
        new_cfg = copy.deepcopy(config)
        for x in flat_lists:
            val = x["values"][i] if len(x["values"]) == max_len else x["values"][0]
            # Navigate to path and set value
            curr = new_cfg
            for step in x["path"][:-1]:
                curr = curr[step]
            curr[x["path"][-1]] = val
        yield new_cfg
