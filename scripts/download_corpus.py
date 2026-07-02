import argparse
import json
import os

from datasets import load_dataset


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
    parser.add_argument("--output", default="corpus/wiki_bn.jsonl")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Limit article count for debugging. Use 0 for the full split.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading {args.dataset} / {args.config} / {args.split}")
    dataset = load_dataset(args.dataset, args.config, split=args.split)

    article_count = 0
    chunk_count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in dataset:
            if args.max_articles and article_count >= args.max_articles:
                break

            text = row.get("text", "")
            for chunk in iter_chunks(text, args.chunk_words, args.overlap_words):
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                chunk_count += 1

            article_count += 1
            if article_count % 1000 == 0:
                print(f"Processed {article_count} articles, wrote {chunk_count} chunks")

    print(f"Saved {chunk_count} chunks from {article_count} articles to {args.output}")


if __name__ == "__main__":
    main()
