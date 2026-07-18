import os
from copy import deepcopy

import torch

_INTEROP_THREADS_CONFIGURED = False


def active_hardware_profile(config):
    """Return the configured hardware profile name, or None."""
    return config.get("runtime", {}).get("hardware_profile")


def resolve_section(config, section_name):
    """Return a config section with the active hardware profile overlaid."""
    section = deepcopy(config.get(section_name, {}))
    profile = active_hardware_profile(config)
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
            raise ValueError(f"Unsupported LLM verifier load_in mode: {mode}")
        return aliases[normalized]

    # Backward-compatible fallback for old config files.
    load_in_4bit = bool(gemma_config.get("load_in_4bit", False))
    load_in_8bit = bool(gemma_config.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ValueError("LLM verifier config cannot set both load_in_4bit and load_in_8bit.")
    if load_in_4bit:
        return "4bit"
    if load_in_8bit:
        return "8bit"
    return "none"


def describe_active_profile(config):
    """Snapshot of resolved settings driven by runtime.hardware_profile."""
    profile = active_hardware_profile(config)
    gemma = resolve_section(config, "gemma")
    rag = resolve_section(config, "rag")
    load_in = None
    if gemma.get("load_in") is not None or gemma.get("load_in_4bit") or gemma.get("load_in_8bit"):
        load_in = resolve_quantization_mode(gemma)
    return {
        "hardware_profile": profile,
        "verifier_model": gemma.get("model_name") or gemma.get("fast_model_name"),
        "fast_verifier_model": gemma.get("fast_model_name") or gemma.get("model_name"),
        "think_verifier_model": gemma.get("think_model_name") or gemma.get("model_name"),
        "model_loader": gemma.get("model_loader"),
        "load_in": load_in,
        "device_map": gemma.get("device_map"),
        "cuda_max_memory": gemma.get("cuda_max_memory"),
        "max_input_tokens": gemma.get("max_input_tokens"),
        "enable_think_pass": gemma.get("enable_think_pass"),
        "exemplar_top_k": gemma.get("exemplar_top_k"),
        "fast_pass_batch_size": gemma.get("fast_pass_batch_size"),
        "think_pass_batch_size": gemma.get("think_pass_batch_size"),
        "max_think_tokens": gemma.get("max_think_tokens"),
        "rag_batch_size": rag.get("batch_size"),
        "rag_query_batch_size": rag.get("query_batch_size"),
        "rag_embedder": rag.get("model_name"),
    }


def validate_config(config):
    profiles = config.get("hardware_profiles", {})
    profile = active_hardware_profile(config)
    if profiles:
        if not profile:
            raise ValueError(
                "runtime.hardware_profile is required when hardware_profiles are defined. "
                "Set it to one of: " + ", ".join(sorted(profiles))
            )
        if profile not in profiles:
            raise ValueError(
                f"Unknown hardware_profile '{profile}'. "
                f"Choose one of: {', '.join(sorted(profiles))}"
            )

    query_mode = resolve_section(config, "rag").get("query_mode")
    if query_mode not in ("prompt", "prompt_response"):
        raise ValueError("rag.query_mode must be 'prompt' or 'prompt_response'.")

    gemma_config = resolve_section(config, "gemma")
    fast_model = gemma_config.get("fast_model_name") or gemma_config.get("model_name")
    think_model = gemma_config.get("think_model_name") or gemma_config.get("model_name")
    if not fast_model:
        raise ValueError(
            "Resolved gemma.fast_model_name (or model_name) is missing. "
            f"Set it under [hardware_profiles.{profile}.gemma] (or [gemma])."
        )
    if not think_model:
        raise ValueError(
            "Resolved gemma.think_model_name (or model_name) is missing. "
            f"Set it under [hardware_profiles.{profile}.gemma] (or [gemma])."
        )
    load_in = resolve_quantization_mode(gemma_config)
    model_loader = gemma_config.get("model_loader", "multimodal_lm")
    if model_loader not in ("multimodal_lm", "causal_lm"):
        raise ValueError("gemma.model_loader must be 'multimodal_lm' or 'causal_lm'.")
    device_map = gemma_config.get("device_map")
    int8_offload = bool(gemma_config.get("llm_int8_enable_fp32_cpu_offload", False))

    if load_in == "4bit" and device_map == "auto":
        raise ValueError(
            'LLM verifier load_in="4bit" cannot use device_map="auto" when CPU/disk '
            'dispatch may occur. Use load_in="8bit" with '
            "llm_int8_enable_fp32_cpu_offload=true for 8GB, or device_map='cuda:0' "
            "for all-GPU loading."
        )
    if load_in == "8bit" and device_map == "auto" and not int8_offload:
        raise ValueError(
            'LLM verifier load_in="8bit" with device_map="auto" requires '
            "llm_int8_enable_fp32_cpu_offload=true."
        )

    decision = config.get("decision", {})
    if "threshold" in decision:
        threshold = float(decision["threshold"])
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("decision.threshold must be between 0 and 1.")


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
