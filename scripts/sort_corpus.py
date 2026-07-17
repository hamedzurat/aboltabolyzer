import argparse
import json
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_utils import describe_active_profile, validate_config
from src.llm_verifier import GemmaVerifier
from src.tui import banner, count_table, done_panel, info, pipeline_progress, warn

BUCKETS = ("wiki", "famous_bn", "idioms", "literal", "grammar", "skip")

BUCKET_ALIASES = {
    "famous": "famous_bn",
    "famous-bn": "famous_bn",
    "famous_bn": "famous_bn",
    "idiom": "idioms",
    "idioms": "idioms",
    "literal": "literal",
    "grammar": "grammar",
    "wiki": "wiki",
    "wikipedia": "wiki",
    "skip": "skip",
    "discard": "skip",
    "none": "skip",
}

GRAMMAR_PATTERNS = (
    "সমাস",
    "সন্ধি",
    "কারক",
    "বিভক্তি",
    "ব্যাসবাক্য",
    "ধাতু",
    "ক্রিয়া",
    "ক্রিয়া",
    "বিশেষ্য",
    "বিশেষণ",
)
IDIOM_PATTERNS = ("ভাবার্থ", "বাগধারা", "প্রবাদ", "লোকোক্তি")
LITERAL_PATTERNS = ("শাব্দিক অর্থ", "আক্ষরিক অর্থ", "literal meaning")
FAMOUS_BN_PATTERNS = (
    "রবীন্দ্রনাথ",
    "নজরুল",
    "শেখ মুজিব",
    "মুজিবনগর",
    "স্বাধীনতা দিবস",
    "বিজয় দিবস",
    "বিজয় দিবস",
    "অপারেশন সার্চলাইট",
    "বাংলাদেশ",
    "ঢাকা বিশ্ববিদ্যালয়",
    "ঢাকা বিশ্ববিদ্যালয়",
    "সুন্দরবন",
)


def text_from_obj(obj, text_keys):
    for key in text_keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_bucket(output):
    normalized = re.sub(r"[^A-Za-z0-9_\-]+", " ", str(output).lower())
    for token in normalized.split():
        if token in BUCKET_ALIASES:
            return BUCKET_ALIASES[token]
    for key, bucket in BUCKET_ALIASES.items():
        if key in normalized:
            return bucket
    return None


def heuristic_bucket(text):
    text = str(text)
    lower = text.lower()
    if any(pattern in text for pattern in GRAMMAR_PATTERNS):
        return "grammar"
    if any(pattern in text for pattern in LITERAL_PATTERNS) or "শাব্দিক" in text:
        return "literal"
    if any(pattern in text for pattern in IDIOM_PATTERNS) or re.search(
        r"[^।:]{2,30}:\s*[^।]+", text
    ):
        if any(word in text for word in ("অর্থ", "বোঝায়", "বুঝায়", "মানে")):
            return "idioms"
    if any(pattern in text for pattern in FAMOUS_BN_PATTERNS):
        return "famous_bn"
    if len(text.split()) >= 5:
        return "wiki"
    if "literal" in lower:
        return "literal"
    return "skip"


def corpus_sort_prompt(text):
    return (
        "Sort this Bangla corpus line into one bucket for a hallucination detector.\n"
        "Buckets:\n"
        "wiki = broad factual encyclopedia facts, not specifically the special buckets\n"
        "famous_bn = Bangladesh history, Bangla literature, famous Bangladeshi/Bengali facts\n"
        "idioms = Bengali idiom/proverb/ভাবার্থ/figurative phrase meanings\n"
        "literal = শাব্দিক or আক্ষরিক word meanings\n"
        "grammar = Bangla grammar rules/examples: সমাস, সন্ধি, কারক, বিভক্তি, ব্যাসবাক্য, ধাতু\n"
        "skip = noisy, malformed, too vague, duplicate header, not useful as evidence\n"
        "Return only one bucket name.\n"
        f"Text:\n{text}\n"
        "bucket:"
    )


class CorpusSorter:
    def __init__(self):
        self.verifier = GemmaVerifier()
        self.loaded = False

    def load(self):
        if not self.loaded:
            self.verifier.load_model()
            self.loaded = True

    def classify(self, text):
        self.load()
        prompt = corpus_sort_prompt(text)
        messages = [{"role": "user", "content": prompt}]
        rendered = self.verifier._apply_chat_template(
            messages,
            enable_thinking=False,
        )
        inputs = self.verifier._prepare_inputs(rendered, use_inputs_embeds=False)
        with torch.inference_mode():
            outputs = self.verifier.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = self.verifier.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return parse_bucket(generated), generated.strip()


def iter_jsonl(path, text_keys):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                yield line_no, {"text": raw}, raw, "invalid_json"
                continue
            if not isinstance(obj, dict):
                yield line_no, {"text": raw}, raw, "not_object"
                continue
            text = text_from_obj(obj, text_keys)
            yield line_no, obj, text, "" if text else "missing_text"


def count_jsonl_rows(path, limit=0):
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit and count >= limit:
                    return count
    return count


class JsonlWriters:
    def __init__(self, output_root, output_name, append, annotate, skipped_path):
        self.output_root = Path(output_root)
        self.output_name = output_name
        self.append = append
        self.annotate = annotate
        self.skipped_path = Path(skipped_path)
        self.handles = {}

    def path_for_bucket(self, bucket):
        if bucket == "skip":
            return self.skipped_path
        return self.output_root / bucket / self.output_name

    def write(self, bucket, obj, meta):
        path = self.path_for_bucket(bucket)
        if bucket not in self.handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.append else "w"
            self.handles[bucket] = open(path, mode, encoding="utf-8")
        output = dict(obj)
        if self.annotate:
            output.update(meta)
        self.handles[bucket].write(json.dumps(output, ensure_ascii=False) + "\n")

    def close(self):
        for handle in self.handles.values():
            handle.close()


def run_sort(args):
    text_keys = [key.strip() for key in args.text_keys.split(",") if key.strip()]
    input_path = Path(args.input)
    output_name = args.output_name or input_path.name
    skipped_path = args.skipped_path or f"generated/corpus_sort_skipped/{input_path.name}"

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    validate_config(config)
    profile = describe_active_profile(config)

    banner("Sort corpus JSONL", f"{input_path} → {args.output_root}/<bucket>/{output_name}")
    info(f"Active verifier: {profile['verifier_model']} ({profile['hardware_profile']})")
    if args.dry_run:
        warn("Dry run: no files will be written")

    sorter = CorpusSorter()
    writers = None
    if not args.dry_run:
        writers = JsonlWriters(
            output_root=args.output_root,
            output_name=output_name,
            append=args.append,
            annotate=args.annotate,
            skipped_path=skipped_path,
        )

    total_rows = count_jsonl_rows(input_path, args.limit)
    counts = Counter()
    invalid_outputs = 0
    try:
        with pipeline_progress() as progress:
            task = progress.add_task("Classifying", total=total_rows)
            for row_idx, (line_no, obj, text, row_error) in enumerate(
                iter_jsonl(input_path, text_keys),
                start=1,
            ):
                if args.limit and row_idx > args.limit:
                    break
                if row_error:
                    bucket = "skip"
                    raw_output = row_error
                else:
                    bucket, raw_output = sorter.classify(text)
                    if bucket is None:
                        invalid_outputs += 1
                        bucket = heuristic_bucket(text) if args.fallback == "heuristic" else "skip"

                if bucket not in BUCKETS:
                    bucket = "skip"
                counts[bucket] += 1

                if writers is not None:
                    meta = {
                        "_sort_bucket": bucket,
                        "_sort_line": line_no,
                        "_sort_model_output": raw_output,
                    }
                    writers.write(bucket, obj, meta)

                progress.update(task, description=f"Classifying · {bucket}")
                progress.advance(task)
    finally:
        if writers is not None:
            writers.close()

    count_table("Sorted corpus lines", {k: int(v) for k, v in counts.items()})
    if invalid_outputs:
        warn(f"LLM returned {invalid_outputs} unparseable bucket(s); fallback={args.fallback}")

    lines = [f"{bucket}: {counts[bucket]}" for bucket in BUCKETS if counts[bucket]]
    if not args.dry_run:
        lines.append(f"Corpus root: {args.output_root}")
        lines.append(f"Skipped: {skipped_path}")
    done_panel("Corpus sort complete", lines or ["No rows processed"])


def tui_args(parser):
    import questionary

    input_path = questionary.path("Input .jsonl file:").ask()
    output_root = questionary.text("Corpus root:", default="corpus").ask()
    output_name = questionary.text(
        "Output filename:",
        default=Path(input_path or "sorted.jsonl").name,
    ).ask()
    dry_run = questionary.confirm("Dry run?", default=True).ask()
    append = False
    if not dry_run:
        append = questionary.confirm("Append to existing files?", default=False).ask()
    limit = questionary.text("Limit rows (blank = all):", default="").ask()

    args = parser.parse_args([])
    args.input = input_path
    args.output_root = output_root or "corpus"
    args.output_name = output_name or Path(input_path).name
    args.dry_run = bool(dry_run)
    args.append = bool(append)
    args.limit = int(limit) if str(limit or "").strip() else 0
    return args


def build_parser():
    parser = argparse.ArgumentParser(
        description="Use the active verifier LLM to sort JSONL lines into typed corpus folders."
    )
    parser.add_argument("input", nargs="?", help="Input .jsonl file")
    parser.add_argument("--config", default="configs/config.toml")
    parser.add_argument("--output-root", default="corpus")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--text-keys", default="text,passage,content")
    parser.add_argument("--skipped-path", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fallback", choices=["heuristic", "skip"], default="heuristic")
    parser.add_argument("--tui", action="store_true", help="Prompt for options interactively")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.tui or not args.input:
        args = tui_args(parser)
    if not args.input:
        parser.error("input .jsonl file is required")
    run_sort(args)


if __name__ == "__main__":
    main()
