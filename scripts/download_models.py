import argparse
import os
import tomllib

from huggingface_hub import snapshot_download

from src.config_utils import resolve_section, validate_config


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
        help="Also download the configured Gemma verifier model. Requires HF access if gated.",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Download XLM-R models from all configured hardware profiles.",
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

    xlmr_config = resolve_section(config, "xlmr")
    model_names = [xlmr_config["model_name"], config["rag"]["model_name"]]

    if args.all_profiles:
        for profile in config.get("hardware_profiles", {}).values():
            if "xlmr" in profile:
                model_names.append(profile["xlmr"].get("model_name"))

    if args.include_gemma:
        gemma_config = resolve_section(config, "gemma")
        model_names.append(gemma_config["model_name"])

    os.makedirs(args.models_dir, exist_ok=True)

    for model_name in unique(model_names):
        local_dir = os.path.join(args.models_dir, model_name.replace("/", "__"))
        print(f"Downloading {model_name} -> {local_dir}")
        snapshot_download(repo_id=model_name, local_dir=local_dir)

    print("Model download step complete.")


if __name__ == "__main__":
    main()
