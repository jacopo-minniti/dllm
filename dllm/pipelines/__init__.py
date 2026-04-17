# Lazy imports to avoid environment issues (like broken Triton/Torch versions)
# during initial package loading.
def __getattr__(name):
    if name in ["a2d", "bert", "dream", "editflow", "fastdllm_v2", "llada", "llada2"]:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ["a2d", "bert", "dream", "editflow", "fastdllm_v2", "llada", "llada2"]
