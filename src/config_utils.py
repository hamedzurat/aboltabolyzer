import os
from copy import deepcopy

import torch


_INTEROP_THREADS_CONFIGURED = False


def resolve_section(config, section_name):
    """Return a config section with the active hardware profile overlaid."""
    section = deepcopy(config.get(section_name, {}))
    profile = config.get("runtime", {}).get("hardware_profile")
    if not profile:
        return section

    profile_overrides = config.get("hardware_profiles", {}).get(profile, {}).get(section_name, {})
    section.update(profile_overrides)
    return section


def fail_on_model_error(config):
    return bool(config.get("runtime", {}).get("fail_on_model_error", True))


def apply_runtime_settings(config):
    global _INTEROP_THREADS_CONFIGURED

    runtime = config.get("runtime", {})

    torch_num_threads = int(runtime.get("torch_num_threads", 0) or 0)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    torch_interop_threads = int(runtime.get("torch_interop_threads", 0) or 0)
    if torch_interop_threads > 0 and not _INTEROP_THREADS_CONFIGURED:
        torch.set_num_interop_threads(torch_interop_threads)
        _INTEROP_THREADS_CONFIGURED = True

    matmul_precision = runtime.get("float32_matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(runtime.get("cuda_benchmark", False))
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(runtime.get("allow_tf32", True))


def resolve_model_path(model_name, models_dir="models/hf"):
    if not model_name:
        return model_name
    # Check if a downloaded snapshot exists under models/hf/
    local_dir = os.path.join(models_dir, model_name.replace("/", "__"))
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        return local_dir
    return model_name
