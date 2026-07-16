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


def resolve_runtime(config):
    return resolve_section(config, "runtime")


def resolve_quantization_mode(gemma_config):
    mode = gemma_config.get("load_in")
    if mode is not None:
        normalized = str(mode).lower().replace("-", "").replace("_", "")
        aliases = {
            "4": "4bit",
            "4bit": "4bit",
            "8": "8bit",
            "8bit": "8bit",
            "none": "none",
            "false": "none",
            "no": "none",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported Gemma load_in mode: {mode}")
        return aliases[normalized]

    # Backward-compatible fallback for old config files.
    load_in_4bit = bool(gemma_config.get("load_in_4bit", False))
    load_in_8bit = bool(gemma_config.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("Gemma config cannot set both load_in_4bit and load_in_8bit.")
    if load_in_4bit:
        return "4bit"
    if load_in_8bit:
        return "8bit"
    return "none"


def validate_config(config):
    profile = config.get("runtime", {}).get("hardware_profile")
    if profile and profile not in config.get("hardware_profiles", {}):
        raise ValueError(f"Unknown hardware_profile '{profile}'.")

    num_folds = int(config.get("num_folds", 0))
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    query_mode = resolve_section(config, "rag").get("query_mode")
    if query_mode not in ("prompt", "prompt_response"):
        raise ValueError("rag.query_mode must be 'prompt' or 'prompt_response'.")

    gemma_config = resolve_section(config, "gemma")
    load_in = resolve_quantization_mode(gemma_config)
    model_loader = gemma_config.get("model_loader", "multimodal_lm")
    if model_loader not in ("multimodal_lm", "causal_lm"):
        raise ValueError("gemma.model_loader must be 'multimodal_lm' or 'causal_lm'.")
    device_map = gemma_config.get("device_map")
    int8_offload = bool(gemma_config.get("llm_int8_enable_fp32_cpu_offload", False))

    if load_in == "4bit" and device_map == "auto":
        raise ValueError(
            'Gemma load_in="4bit" cannot use device_map="auto" when CPU/disk '
            'dispatch may occur. Use load_in="8bit" with '
            "llm_int8_enable_fp32_cpu_offload=true for 8GB, or device_map='cuda:0' "
            "for all-GPU loading."
        )
    if load_in == "8bit" and device_map == "auto" and not int8_offload:
        raise ValueError(
            'Gemma load_in="8bit" with device_map="auto" requires '
            "llm_int8_enable_fp32_cpu_offload=true."
        )

    xlmr_config = resolve_section(config, "xlmr")
    if int(xlmr_config.get("batch_size", 0)) < 1:
        raise ValueError("xlmr.batch_size must be at least 1.")
    if int(xlmr_config.get("max_length", 0)) < 1:
        raise ValueError("xlmr.max_length must be at least 1.")


def fail_on_model_error(config):
    return bool(resolve_runtime(config).get("fail_on_model_error", True))


def use_llm_verifier(config):
    return bool(resolve_runtime(config).get("use_llm_verifier", True))


def apply_runtime_settings(config):
    global _INTEROP_THREADS_CONFIGURED

    runtime = resolve_runtime(config)

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
