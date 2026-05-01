import argparse
import ast
import glob
import os
import shlex
import shutil
import subprocess
import sys
import copy
import yaml

from dllm.utils.naming import get_experiment_naming, flatten_config_dict
from dllm.utils.config import load_resolved_config, resolve_keywords, expand_matrix_config

_DLLM_BASE_CONFIGS = "dllm/utils/configs.py"
_EXAMPLE_SCRIPTS_GLOB = "examples/**/*.py"


def resolve_config_path(path: str, base_dir: str) -> str:
    if os.path.exists(path):
        return path
    p = os.path.join(base_dir, path)
    if os.path.exists(p):
        return p
    if not p.endswith(".yaml") and os.path.exists(p + ".yaml"):
        return p + ".yaml"
    return path


def load_dotenv(path: str = ".env") -> dict:
    """Parse a .env file and return key→value pairs (no subprocess needed)."""
    env = {}
    if not os.path.exists(path):
        return env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            line = line.removeprefix("export").strip()
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                val = val[1:-1]
            elif "#" in val:
                val = val[: val.index("#")].strip()
            env[key] = val
    return env


def _parse_dataclass_fields(filepath: str) -> dict[str, set[str]]:
    """Return {ClassName: {field_names}} for all @dataclass classes in a file."""
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)
        result = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_dc = any(
                (isinstance(d, ast.Name) and d.id == "dataclass") or
                (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                for d in node.decorator_list
            )
            if not is_dc:
                continue
            fields = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.add(item.target.id)
            result[node.name] = fields
        return result
    except Exception:
        return {}


def _hfparser_class_names(filepath: str) -> list[str]:
    """Return the class names passed to HfArgumentParser in a script file."""
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_hf = (
                (isinstance(func, ast.Name) and "HfArgumentParser" in func.id) or
                (isinstance(func, ast.Attribute) and func.attr == "HfArgumentParser")
            )
            if is_hf and node.args:
                first = node.args[0]
                if isinstance(first, ast.Tuple):
                    return [e.id for e in first.elts if isinstance(e, ast.Name)]
                if isinstance(first, ast.Name):
                    return [first.id]
    except Exception:
        pass
    return []


def get_script_rejected_args(target_script: str) -> set:
    """
    Return the set of arg names to SKIP when launching target_script.

    An arg is rejected when it is defined in some other example script's
    local dataclasses (i.e. it's a real per-script arg, not a generic
    transformers.TrainingArguments field) but absent from target_script's
    local dataclasses.  Generic transformers args (gradient_checkpointing,
    bf16, …) are not defined in any local dataclass and always pass through.
    """
    # Fields shared by all scripts via dllm.utils base classes
    base_fields: set[str] = set()
    if os.path.exists(_DLLM_BASE_CONFIGS):
        for fields in _parse_dataclass_fields(_DLLM_BASE_CONFIGS).values():
            base_fields |= fields

    # Collect per-script local fields for every example script
    all_script_fields: set[str] = set()
    target_fields: set[str] = set()

    for path in glob.glob(_EXAMPLE_SCRIPTS_GLOB, recursive=True):
        classes = _parse_dataclass_fields(path)
        parser_classes = _hfparser_class_names(path)
        local_fields: set[str] = set()
        for cls in parser_classes:
            if cls in classes:
                local_fields |= classes[cls]
        all_script_fields |= local_fields
        if os.path.normpath(path) == os.path.normpath(target_script):
            target_fields |= local_fields

    # Reject only args that appear in some script but not in this one
    return all_script_fields - target_fields - base_fields


def main():
    parser = argparse.ArgumentParser(description="Local Training Launcher for dLLM")

    parser.add_argument("run_config", help="Name or path of training/wandb configuration YAML")
    parser.add_argument("session", help="Name of the tmux session to create/use")
    parser.add_argument(
        "--accelerate_config",
        default="fsdp",
        help="Name of accelerate config (located in configs/accelerate/)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs (default: read from accelerate config, fallback 4)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Generate scripts but do not launch",
    )
    parser.add_argument(
        "--no_tmux",
        action="store_true",
        help="Run directly in the current terminal (blocking) instead of tmux",
    )

    args, extra_args = parser.parse_known_args()

    # --- Load .env early so values are available for naming / wandb project ---
    dotenv = load_dotenv(".env")
    for k, v in dotenv.items():
        os.environ.setdefault(k, v)

    # --- Keyword map ---
    keyword_map = {}
    if os.path.exists("configs/keywords.yaml"):
        with open("configs/keywords.yaml") as f:
            keyword_map = yaml.safe_load(f) or {}

    # --- Resolve accelerate config path ---
    acc_config = args.accelerate_config
    if not acc_config.endswith(".yaml"):
        acc_config = f"configs/accelerate/{acc_config}.yaml"

    # --- Determine num_gpus (CLI > accelerate config > fallback 4) ---
    num_gpus = args.num_gpus
    if num_gpus is None:
        if os.path.exists(acc_config):
            with open(acc_config) as f:
                acc_cfg = yaml.safe_load(f) or {}
            num_gpus = int(acc_cfg.get("num_processes", 4))
        else:
            num_gpus = 4

    # --- Load & resolve training config ---
    base_run_cfg = load_resolved_config(args.run_config, "configs/train", "../default.yaml")
    if "training" in base_run_cfg:
        base_run_cfg["training"] = flatten_config_dict(base_run_cfg["training"])

    # --- Apply --training.key=val CLI overrides ---
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--training."):
            if "=" in arg:
                key_full, val = arg.split("=", 1)
            else:
                key_full = arg
                val = extra_args[i + 1] if i + 1 < len(extra_args) else True
                i += 1
            _, key = key_full.lstrip("-").split(".", 1)
            if isinstance(val, str):
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.startswith("[") and val.endswith("]"):
                    try:
                        val = ast.literal_eval(val)
                    except Exception:
                        pass
            if "training" not in base_run_cfg:
                base_run_cfg["training"] = {}
            base_run_cfg["training"][key] = val
        i += 1

    base_run_cfg = resolve_keywords(base_run_cfg, keyword_map)

    try:
        exp_configs = list(expand_matrix_config(base_run_cfg))
    except ValueError as e:
        print(f"❌ Configuration Matrix Error: {e}")
        sys.exit(1)

    print(f"🧪 Found {len(exp_configs)} experiment(s).")
    print(f"🖥️  Using {num_gpus} GPU(s)  |  accelerate config: {acc_config}")

    generated_scripts = []

    for idx, run_cfg in enumerate(exp_configs):
        if "training" not in run_cfg:
            run_cfg["training"] = {}
        if "seed" not in run_cfg["training"]:
            run_cfg["training"]["seed"] = 42

        group, run_name, tags, output_dir = get_experiment_naming(
            run_cfg, {"job_name": args.session}
        )

        training = run_cfg["training"]
        script_path = training.pop("script_path", "examples/llada/sft.py")
        training["output_dir"] = output_dir

        # Copy local model .py files if model path is a local directory
        if not args.dry_run and "model_name_or_path" in training:
            model_src = training["model_name_or_path"]
            if os.path.exists(model_src) and os.path.isdir(model_src):
                print(f"ℹ️  Copying model files from {model_src} → {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                for fname in os.listdir(model_src):
                    if fname.endswith(".py"):
                        shutil.copy2(os.path.join(model_src, fname), output_dir)

        # --- AST-based arg filtering (instant, no subprocess) ---
        rejected_args = get_script_rejected_args(script_path)
        if rejected_args:
            print(f"ℹ️  [{run_name}] Filtering args not in {script_path}: {', '.join(sorted(rejected_args))}")

        # --- Build training flags ---
        train_flags = []
        skipped_args = []
        for k, v in training.items():
            if k in rejected_args:
                skipped_args.append(k)
                continue
            if v is True:
                train_flags.append(f"--{k}")
            elif v is False:
                train_flags.append(f"--{k} False")
            elif v is None:
                continue
            else:
                if isinstance(v, list):
                    v = " ".join(map(str, v))
                train_flags.append(f"--{k} {v}")

        # Pass-through extra CLI args that are not --training.* prefixed
        for arg in extra_args:
            if not arg.startswith("--training."):
                train_flags.append(arg)

        # --- Build accelerate launch flags ---
        launch_parts = [f"--num_processes {num_gpus}"]
        if num_gpus > 1:
            launch_parts.append("--multi_gpu")
        if "fsdp" in acc_config.lower():
            launch_parts.append(
                "--fsdp_transformer_layer_cls_to_wrap LLaDABlock,Fast_dLLM_QwenDecoderLayer"
            )
        launch_flags = " ".join(launch_parts)

        # --- Build env exports block ---
        env_lines = []

        # 1. Everything from .env (already loaded in Python; re-export so child
        #    processes spawned by accelerate also inherit them)
        for k, v in dotenv.items():
            env_lines.append(f'export {k}="{v}"')

        # 2. Standard dLLM runtime vars
        env_lines += [
            "export HF_HOME=\"$PWD/.cache\"",
            "export HF_DATASETS_CACHE=\"$PWD/.cache/datasets\"",
            "export PYTHONHASHSEED=42",
            "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=43200",
            "export ACCELERATE_TIMEOUT_IN_SECONDS=43200",
            "export WANDB_INIT_TIMEOUT=300",
        ]

        # 3. Experiment-specific wandb vars (overrides any .env values)
        env_lines += [
            f'export WANDB_NAME="{run_name}"',
            f'export WANDB_RUN_GROUP="{group}"',
            f'export WANDB_TAGS="{",".join(tags)}"',
            f'export WANDB_PROJECT="{os.environ.get("WANDB_PROJECT", "BPTT-llada")}"',
        ]

        env_block = "\n".join(env_lines)
        train_args_str = " \\\n    ".join(train_flags)

        # --- Write generated shell script ---
        cwd_quoted = shlex.quote(os.getcwd())
        script_content = f"""#!/bin/bash
set -e
cd {cwd_quoted}

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Source .env so shell builtins (e.g. conda activate) also work
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

# Runtime and experiment env vars
{env_block}

echo -e "\\n\\033[1;32m--- Experiment {idx + 1}/{len(exp_configs)}: {run_name} ---\\033[0m"

accelerate launch {launch_flags} \\
    --config_file "{acc_config}" \\
    "{script_path}" \\
    {train_args_str}
"""
        if len(exp_configs) > 1:
            script_name = f".generated_train_local_{idx}.sh"
        else:
            script_name = ".generated_train_local.sh"

        with open(script_name, "w") as f:
            f.write(script_content)
        os.chmod(script_name, 0o755)
        generated_scripts.append((script_name, run_name))

        print(f"📝 Generated: {script_name}  ({run_name})")
        if args.dry_run:
            print("─" * 70)
            print(script_content)
            print("─" * 70)

    if args.dry_run:
        print("🔍 [Dry Run] No jobs launched.")
        return

    os.makedirs(".logs", exist_ok=True)

    if args.no_tmux:
        # Run sequentially in the current terminal (blocking)
        if len(generated_scripts) > 1:
            print("⚠️  Multiple experiments — running sequentially.")
        for script_name, run_name in generated_scripts:
            print(f"🚀 Launching: {run_name}")
            result = subprocess.run(["bash", script_name])
            if result.returncode != 0:
                print(f"❌ Experiment failed: {run_name}")
                sys.exit(result.returncode)
        return

    # --- Launch via tmux ---
    session = args.session
    tmux_check = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)

    if tmux_check.returncode == 0:
        print(f"⚠️  Session '{session}' already exists — opening new window 'train'.")
        subprocess.run(["tmux", "new-window", "-t", session, "-n", "train"])
        target = f"{session}:train"
    else:
        subprocess.run(["tmux", "new-session", "-d", "-s", session])
        target = session

    # Chain experiments: bash s0.sh && bash s1.sh && ...
    chain = " && bash ".join(s for s, _ in generated_scripts)
    full_cmd = f"bash {chain}; echo -e '\\n\\033[1;32m✅ All experiments done.\\033[0m'"
    subprocess.run(["tmux", "send-keys", "-t", target, full_cmd, "C-m"])

    print(f"\n🚀 Launched {len(generated_scripts)} experiment(s) in tmux session '{session}'.")
    print(f"   Attach:  tmux attach -t {session}")


if __name__ == "__main__":
    main()
