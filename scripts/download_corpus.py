import argparse
import json
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tui import banner, count_table, done_panel, info, ok, pipeline_progress

# Default keywords targeting people, places, history, literature, and prominent concepts

DEFAULT_KEYWORDS = [
    # # People & Occupations
    # "জন্ম",
    # "মৃত্যু",
    # "জন্মগ্রহণ",
    # "কবি",
    # "লেখক",
    # "নেতা",
    # "রাজনীতিবিদ",
    # "অভিনেতা",
    # "বিজ্ঞানী",
    # "সম্রাট",
    # "ব্যক্তিত্ব",
    # # Places & Geography
    # "অবস্থিত",
    # "জেলা",
    # "বিভাগ",
    # "নদী",
    # "পাহাড়",
    # "শহর",
    # "রাজধানী",
    # "উপজেলা",
    # "গ্রাম",
    # "দেশ",
    # "সাগর",
    # "মহাসাগর",
    # # History, Events, Institutions
    # "যুদ্ধ",
    # "প্রতিষ্ঠিত",
    # "খ্রিস্টাব্দ",
    # "সালে",
    # "ইতিহাস",
    # "আন্দোলন",
    # "বিশ্ববিদ্যালয়",
    # "সংসদ",
    # "সংবিধান",
    # "প্রতিষ্ঠা",
    # # Literature, Art & Culture
    # "উপন্যাস",
    # "রচিত",
    # "কবিতা",
    # "নাটক",
    # "গল্প",
    # "গান",
    # "চলচ্চিত্র",
    # "পত্রিকা",
    # "সাময়িকী",
    # "ভাষা",
    # "ভাবার্থ",
    # "অভিধান",
    # People's name
    "ঈশ্বরচন্দ্র বিদ্যাসাগর",
    "রামমোহন রায়",
    "বঙ্কিমচন্দ্র চট্টোপাধ্যায়",
    "শরৎচন্দ্র চট্টোপাধ্যায়",
    "মাইকেল মধুসূদন দত্ত",
    "কাজী নজরুল ইসলাম",
    "জসীমউদ্দীন",
    "জীবনানন্দ দাশ",
    "সুকান্ত ভট্টাচার্য",
    "তারাশঙ্কর বন্দ্যোপাধ্যায়",
    "মানিক বন্দ্যোপাধ্যায়",
    "বিভূতিভূষণ বন্দ্যোপাধ্যায়",
    "হুমায়ূন আহমেদ",
    "সুনীল গঙ্গোপাধ্যায়",
    "শক্তি চট্টোপাধ্যায়",
    "শঙ্খ ঘোষ",
    "আল মাহমুদ",
    "জয়নুল আবেদীন",
]


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
    parser.add_argument("--output", default="generated/wiki_filtered.jsonl")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Limit article count for debugging. Use 0 for the full split.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=True,
        help="Filter articles using the keyword list.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_false",
        dest="filter",
        help="Disable keyword filtering and download the full corpus.",
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="Comma-separated list of additional keywords/names to include in the filter.",
    )
    args = parser.parse_args()

    # Build final list of keywords
    keywords = list(DEFAULT_KEYWORDS)
    if args.keywords:
        extra_keys = [k.strip() for k in args.keywords.split(",") if k.strip()]
        keywords.extend(extra_keys)
        info(f"Added {len(extra_keys)} extra custom keywords: {extra_keys}")

    banner(
        "Download wiki corpus",
        f"{args.dataset} / {args.config} → {args.output}",
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    info(f"Loading dataset split '{args.split}'...")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    total = len(dataset) if args.max_articles <= 0 else min(len(dataset), args.max_articles)
    info(f"Will process up to {total} articles")

    processed_count = 0
    kept_articles = 0
    skipped_articles = 0
    chunk_count = 0
    keyword_counts = {kw: 0 for kw in keywords}

    with open(args.output, "w", encoding="utf-8") as f:
        with pipeline_progress() as progress:
            task = progress.add_task("Chunking wiki", total=total)
            for row in dataset:
                if args.max_articles and processed_count >= args.max_articles:
                    break

                title = row.get("title", "")
                text = row.get("text", "")

                # Apply keyword filtering if enabled
                if args.filter:
                    combined_text = f"{title} {text}"
                    matched_kws = [kw for kw in keywords if kw in combined_text]
                    for kw in matched_kws:
                        keyword_counts[kw] += 1

                    if not matched_kws:
                        skipped_articles += 1
                        processed_count += 1
                        progress.advance(task)
                        continue

                kept_articles += 1
                for chunk in iter_chunks(text, args.chunk_words, args.overlap_words):
                    f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                    chunk_count += 1

                processed_count += 1
                if processed_count % 100 == 0 or processed_count == total:
                    progress.update(
                        task,
                        description=f"Chunking wiki · {chunk_count} chunks",
                    )
                progress.advance(task)

    ok(f"Wrote {chunk_count} chunks from {kept_articles} articles (skipped {skipped_articles})")

    if args.filter:
        count_table("Keyword Matches", keyword_counts, key_header="Keyword", limit=len(keywords))

    done_panel(
        "Corpus ready",
        [
            f"Articles processed: {processed_count}",
            f"Articles kept:      {kept_articles}",
            f"Articles skipped:   {skipped_articles}",
            f"Total chunks:       {chunk_count}",
            f"Output:             {args.output}",
        ],
    )


if __name__ == "__main__":
    main()
