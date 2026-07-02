import os
from copy import deepcopy


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


def resolve_model_path(model_name, models_dir="models/hf"):
    if not model_name:
        return model_name
    # Check if a downloaded snapshot exists under models/hf/
    local_dir = os.path.join(models_dir, model_name.replace("/", "__"))
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        return local_dir
    return model_name
