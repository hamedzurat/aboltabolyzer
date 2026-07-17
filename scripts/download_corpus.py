import argparse
import json
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tui import banner, done_panel, info, ok, pipeline_progress


def iter_chunks(text, chunk_words, overlap_words):
    words = str(text).split()
    if not words:
        return

    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if len(chunk) >= 50:
            yield chunk


def main():
    parser = argparse.ArgumentParser(description="Download and chunk Bengali Wikipedia corpus.")
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--config", default="20231101.bn")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="corpus/wiki/wiki_bn.jsonl")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Limit article count for debugging. Use 0 for the full split.",
    )
    args = parser.parse_args()

    banner(
        "Download wiki corpus",
        f"{args.dataset} / {args.config} → {args.output}",
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    info(f"Loading dataset split '{args.split}'...")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    total = len(dataset) if args.max_articles <= 0 else min(len(dataset), args.max_articles)
    info(f"Will process up to {total} articles")

    article_count = 0
    chunk_count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        with pipeline_progress() as progress:
            task = progress.add_task("Chunking wiki", total=total)
            for row in dataset:
                if args.max_articles and article_count >= args.max_articles:
                    break

                text = row.get("text", "")
                for chunk in iter_chunks(text, args.chunk_words, args.overlap_words):
                    f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                    chunk_count += 1

                article_count += 1
                if article_count % 100 == 0 or article_count == total:
                    progress.update(
                        task,
                        description=f"Chunking wiki · {chunk_count} chunks",
                    )
                progress.advance(task)

    ok(f"Wrote {chunk_count} chunks from {article_count} articles")
    done_panel(
        "Corpus ready",
        [
            f"Articles: {article_count}",
            f"Chunks: {chunk_count}",
            f"Output: {args.output}",
        ],
    )


if __name__ == "__main__":
    main()
