# ruff: noqa: E402
import argparse
import json
import os
import re
import sys
import time
import tomllib
import warnings
from collections import Counter
from pathlib import Path

# Suppress PyTorch/bitsandbytes FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_utils import describe_active_profile, validate_config
from src.llm_verifier import GemmaVerifier
from src.tui import banner, count_table, done_panel, info, pipeline_progress, warn

BUCKETS = ("wiki", "idioms", "literal", "grammar", "skip")

BUCKET_ALIASES = {
    "famous": "wiki",
    "famous-bn": "wiki",
    "famous_bn": "wiki",
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


def corpus_sort_prompt(text):
    # Truncate text to avoid token limits on very long Wikipedia pages
    truncated_text = str(text)[:1500]
    return (
        "You are an expert data cleaner and annotator. Your goal is to convert the input text into a list of clean, sorted, atomic, and self-contained factual statements in Bengali, and classify them into their correct category.\n\n"
        "Instructions for Rewording:\n"
        "1. Resolve Pronouns & Context: Replace vague words (e.g., 'তিনি', 'ওনার', 'এটি', 'সেখানে') with the actual names, entities, or concepts they refer to. Every generated fact must be fully understandable on its own without needing external context.\n"
        "2. Prepend Story/Book Context: If the text is from a story, book, or poem (like 'তোতা-কাহিনি' or 'রবীন্দ্রনাথ ঠাকুরের কবিতা'), explicitly add this context to the fact (e.g., 'রবীন্দ্রনাথ ঠাকুরের তোতা-কাহিনি গল্পে রাজা পাখিটিকে শিক্ষা দেওয়ার নির্দেশ দেন।').\n"
        "3. Convert Questions/Blanks: Turn questions (e.g., 'পদ্মা নদীর শাখা নদী কোনটি?') and fill-in-the-blanks (e.g., 'বাংলাদেশ ______ সালে অলিম্পিকে অংশ নেয়') into direct, declarative facts (e.g., 'পদ্মা নদীর শাখা নদী হলো গড়াই ও ধলেশ্বরী।' or 'বাংলাদেশ ১৯৮৪ সালে অলিম্পিকে অংশগ্রহণ করে।').\n"
        "4. Handle Long/Messy Paragraphs: If the input is a long article or contains multiple paragraphs, extract all key factual statements (up to 5-10 facts). Split them so each fact is a single, short, atomic sentence.\n"
        "5. Output Format: Return a list of reworded facts, with one fact per line prefixed by a dash (-). If it is a clean single fact, output just that one fact.\n\n"
        "Categories:\n"
        "- wiki: General facts (science, global history/geography, global events), Bangladesh history, Bangla literature, Bangladeshi geography, famous Bengali personalities/events, and Bengali literary stories (e.g. তোতা-কাহিনি)\n"
        "- idioms: Bengali idioms, proverbs, or figurative meanings (ভাবার্থ, বাগধারা)\n"
        "- literal: Word translation or literal meanings (শাব্দিক অর্থ, আক্ষরিক অর্থ)\n"
        "- grammar: Bangla grammar rules, definitions, and grammatical examples (সমাস, সন্ধি, কারক)\n"
        "- skip: Math/equations/numbers, English language, questions, code, noisy or low-quality text\n\n"
        "Output Format:\n"
        "bucket: <category>\n"
        "reworded:\n"
        "- <Factual Sentence 1>\n"
        "- <Factual Sentence 2> (if multiple)\n\n"
        f"Input Text: {truncated_text}\n"
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
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = self.verifier.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        # Strip think block if present (DeepSeek-R1 models write reasoning inside <think>...</think>)
        cleaned_generated = re.sub(
            r"<think>.*?(?:</think>|$)", "", generated, flags=re.DOTALL
        ).strip()
        # Strip markdown code blocks/fences (e.g. ```yaml or ```)
        cleaned_generated = re.sub(r"```[A-Za-z0-9_-]*", "", cleaned_generated).strip()

        bucket = None
        reworded = text

        bucket_match = re.search(r"bucket\s*[:=]\s*(\w+)", cleaned_generated, re.IGNORECASE)
        if bucket_match:
            bucket = parse_bucket(bucket_match.group(1))
        else:
            bucket = parse_bucket(cleaned_generated)

        text_match = re.search(
            r"reworded\s*[:=]\s*(.+)", cleaned_generated, re.IGNORECASE | re.DOTALL
        )
        if not text_match:
            text_match = re.search(
                r"text\s*[:=]\s*(.+)", cleaned_generated, re.IGNORECASE | re.DOTALL
            )
        if text_match:
            reworded = text_match.group(1).strip()
            # Remove any trailing incomplete UTF-8 characters decoded as \ufffd
            reworded = reworded.replace("\ufffd", "").strip()

        return bucket, reworded, generated.strip()


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
        self.last_flush_time = time.time()

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

        current_time = time.time()
        if current_time - self.last_flush_time >= 60.0:
            for handle in self.handles.values():
                handle.flush()
            self.last_flush_time = current_time

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

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "sort_corpus_debug.jsonl"
    log_mode = "a" if args.append else "w"
    log_file = open(log_path, log_mode, encoding="utf-8")
    last_log_flush = time.time()

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

                fact_lines = []
                if row_error:
                    bucket = "skip"
                    reworded = text
                    raw_output = row_error
                else:
                    bucket, reworded, raw_output = sorter.classify(text)
                    if bucket is None:
                        invalid_outputs += 1
                        bucket = "skip"

                if bucket not in BUCKETS:
                    bucket = "skip"
                counts[bucket] += 1

                # Extract fact lines if valid
                if not row_error and bucket != "skip" and reworded:
                    fact_lines = [
                        f.strip()
                        for f in reworded.split("\n")
                        if f.strip()
                        and not f.strip().lower().startswith("reworded")
                        and not f.strip().lower().startswith("bucket")
                        and not f.strip().lower().startswith("category")
                    ]
                    # Clean bullet points/numbered prefixes
                    fact_lines = [re.sub(r"^[-*•\d.]+\s*", "", f).strip() for f in fact_lines]
                    fact_lines = [f for f in fact_lines if f]

                # Write detailed logs
                log_entry = {
                    "line_no": line_no,
                    "input_text": text,
                    "parsed_bucket": bucket,
                    "reworded_facts": fact_lines if fact_lines else [reworded],
                    "raw_model_output": raw_output,
                    "error": row_error or None,
                }
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                current_time = time.time()
                if current_time - last_log_flush >= 60.0:
                    log_file.flush()
                    last_log_flush = current_time

                if writers is not None:
                    meta = {
                        "_sort_bucket": bucket,
                        "_sort_line": line_no,
                        "_sort_model_output": raw_output,
                    }
                    if bucket != "skip" and fact_lines:
                        for fact in fact_lines:
                            obj_to_write = dict(obj)
                            for key in text_keys:
                                if key in obj_to_write:
                                    obj_to_write[key] = fact
                                    break
                            writers.write(bucket, obj_to_write, meta)
                    else:
                        writers.write(bucket, obj, meta)

                gpu_status = ""
                if torch.cuda.is_available():
                    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                    gpu_status = f" | GPU VRAM: {allocated_gb:.1f}/{reserved_gb:.1f}GB"

                stats_str = " | ".join(f"{b}: {counts[b]}" for b in BUCKETS if counts[b] > 0)
                desc = (
                    f"Classifying | {stats_str}{gpu_status}"
                    if stats_str
                    else f"Classifying{gpu_status}"
                )

                progress.update(task, description=desc)
                progress.advance(task)
    finally:
        log_file.close()
        if writers is not None:
            writers.close()

    count_table("Sorted corpus lines", {k: int(v) for k, v in counts.items()})
    if invalid_outputs:
        warn(f"LLM returned {invalid_outputs} unparseable bucket(s)")

    lines = [f"{bucket}: {counts[bucket]}" for bucket in BUCKETS if counts[bucket]]
    if not args.dry_run:
        lines.append(f"Corpus root: {args.output_root}")
        lines.append(f"Skipped: {skipped_path}")
    done_panel("Corpus sort complete", lines or ["No rows processed"])


def tui_args(parser):
    import questionary

    input_path = questionary.path("Input .jsonl file:").ask()
    if input_path and os.path.exists(input_path):
        try:
            total_rows = count_jsonl_rows(input_path)
            info(f"Selected file: {Path(input_path).name} ({total_rows} rows found)")
        except Exception as e:
            warn(f"Could not read file stats: {e}")

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
    parser.add_argument(
        "--fallback",
        choices=["skip"],
        default="skip",
        help="Fallback option when LLM classification fails (default: skip)",
    )
    parser.add_argument("--tui", action="store_true", help="Prompt for options interactively")
    return parser


def main():
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = build_parser()
    args = parser.parse_args()
    if args.tui or not args.input:
        args = tui_args(parser)
    if not args.input:
        parser.error("input .jsonl file is required")
    run_sort(args)


if __name__ == "__main__":
    main()
