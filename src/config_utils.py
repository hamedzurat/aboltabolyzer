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
