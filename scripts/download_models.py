import argparse
import os
import sys
import tomllib

from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_utils import describe_active_profile, resolve_section, validate_config
from src.tui import banner, done_panel, info, kv_table, ok


def unique(items):
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models used by the pipeline."
    )
    parser.add_argument(
        "--include-gemma",
        action="store_true",
        help="Also download the active configured LLM verifier model.",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="No-op for embedder (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--all-profile-gemmas",
        action="store_true",
        help=(
            "When --include-gemma is set, also download verifier models from all "
            "hardware profiles. Some profiles may require gated HF access."
        ),
    )
    parser.add_argument(
        "--models-dir",
        default="models/hf",
        help="Local directory where model snapshots should be stored.",
    )
    args = parser.parse_args()

    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)
    validate_config(config)

    profile = describe_active_profile(config)
    banner("Download models", f"hardware_profile={profile['hardware_profile']}")
    kv_table(
        "Active profile",
        {
            "hardware_profile": profile["hardware_profile"],
            "verifier_model": profile["verifier_model"],
            "load_in": profile["load_in"],
            "rag_embedder": profile["rag_embedder"],
        },
    )

    model_names = [config["rag"]["model_name"]]

    if args.include_gemma:
        gemma_config = resolve_section(config, "gemma")
        model_names.append(gemma_config["model_name"])
        if args.all_profile_gemmas:
            for name, profile_cfg in config.get("hardware_profiles", {}).items():
                profile_gemma = dict(config.get("gemma", {}))
                profile_gemma.update(profile_cfg.get("gemma", {}))
                model_names.append(profile_gemma.get("model_name"))
                info(f"Also queued profile '{name}' verifier: {profile_gemma.get('model_name')}")

    names = unique(model_names)
    info(f"Will download {len(names)} model(s) → {args.models_dir}")
    os.makedirs(args.models_dir, exist_ok=True)

    for i, model_name in enumerate(names, start=1):
        local_dir = os.path.join(args.models_dir, model_name.replace("/", "__"))
        info(f"[{i}/{len(names)}] {model_name}")
        snapshot_download(repo_id=model_name, local_dir=local_dir, token=False)
        ok(f"Saved → {local_dir}")

    done_panel(
        "Models ready",
        [
            f"Profile: {profile['hardware_profile']}",
            f"Directory: {args.models_dir}",
            f"Count: {len(names)}",
        ],
    )


if __name__ == "__main__":
    main()
