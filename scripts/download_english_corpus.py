import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tui import banner, done_panel, info, ok, pipeline_progress

# Configuration Constants
INPUT_DIR = "generated/wiki"
OUTPUT_DIR = "generated/wiki_en"
ENABLED_CATEGORIES = [
    "writers_poets",
    "politicians_leaders",
    "scientists_scholars",
    "artists_performers",
    "athletes",
    "other_people",
    "countries_states",
    "cities_towns",
    "administrative_units",
    "villages",
    "water_bodies",
    "mountains_islands",
    "other_places",
    "wars_battles",
    "historical_events_revolutions",
    "books_literature",
    "movies_drama",
    "music_songs",
    "art_festivals",
    "educational_institutions",
    "companies_businesses",
    "government_bodies",
    "clubs_associations",
    "religious_texts_doctrines",
    "deities_figures",
    "religious_places",
    "animals",
    "plants",
    "mathematics",
    "physics_chemistry",
    "astronomy_space",
    "technology_computing",
    "medicine_biology",
    "other",
]
MAX_ARTICLES_PER_CATEGORY = 0  # Limit per category for debugging; use 0 for all.
MAX_WORKERS = 8
CHUNK_WORDS = 220
OVERLAP_WORDS = 50

# Use a browser User-Agent to avoid early Wikipedia API blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

BANGLADESH_KEYWORDS = [
    "bangladesh",
    "bangladeshi",
    "bengal",
    "bengali",
    "dhaka",
    "chittagong",
    "sylhet",
    "rajshahi",
    "khulna",
    "barisal",
    "mymensingh",
    "rangpur",
    "comilla",
    "east pakistan",
    "east bengal",
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


def get_english_titles_batch(bn_titles):
    """Finds corresponding English Wikipedia titles for a batch of Bengali titles (max 50)."""
    if not bn_titles:
        return {}

    params = {
        "action": "query",
        "prop": "langlinks",
        "lllang": "en",
        "titles": "|".join(bn_titles),
        "format": "json",
    }
    url = f"https://bn.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    backoff = 1.0

    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                results = {}
                for page in pages.values():
                    title_bn = page.get("title")
                    if "langlinks" in page and title_bn:
                        results[title_bn] = page["langlinks"][0]["*"]
                return results
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
        except Exception:
            break
    return {}


def filter_english_pages_related_batch(en_titles):
    """Checks which English titles in a batch (max 50) are Bangladesh-related."""
    if not en_titles:
        return []

    related = []
    to_query = []

    # 1. Quick check on title strings first (saves API calls)
    for title in en_titles:
        if any(kw in title.lower() for kw in BANGLADESH_KEYWORDS):
            related.append(title)
        else:
            to_query.append(title)

    if not to_query:
        return related

    # 2. Batch check categories for the remaining pages
    params = {
        "action": "query",
        "prop": "categories",
        "cllimit": "50",
        "titles": "|".join(to_query),
        "format": "json",
    }
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    backoff = 1.0

    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    en_title = page.get("title")
                    if "categories" in page and en_title:
                        for cat in page["categories"]:
                            cat_title = cat.get("title", "").lower()
                            if any(kw in cat_title for kw in BANGLADESH_KEYWORDS):
                                related.append(en_title)
                                break
                return related
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
        except Exception:
            break

    return related


def get_english_content(en_title):
    """Fetches the plain text content of a single English Wikipedia article with retry logic."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": en_title,
        "format": "json",
    }
    url = f"https://en.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    backoff = 1.0

    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if "extract" in page:
                        return page["extract"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
        except Exception:
            break
    return None


def main():
    banner(
        "Download English counterparts",
        f"Granular categories → {OUTPUT_DIR}/en_<category>.jsonl",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_summary = []
    total_chunks = 0

    for cat in ENABLED_CATEGORIES:
        titles_file = os.path.join(INPUT_DIR, f"titles_{cat}.txt")
        output_file = os.path.join(OUTPUT_DIR, f"en_{cat}.jsonl")
        output_titles_file = os.path.join(OUTPUT_DIR, f"titles_en_{cat}.txt")

        if not os.path.exists(titles_file):
            continue

        with open(titles_file, "r", encoding="utf-8") as f:
            bn_titles = [line.strip() for line in f if line.strip()]

        if MAX_ARTICLES_PER_CATEGORY > 0:
            bn_titles = bn_titles[:MAX_ARTICLES_PER_CATEGORY]

        if not bn_titles:
            continue

        info(f"Processing '{cat}' ({len(bn_titles)} articles)...")

        # Step 1: Batch translate Bengali titles to English (50 at a time)
        en_titles_map = {}
        with pipeline_progress() as progress:
            task = progress.add_task(f"Resolving {cat}", total=len(bn_titles))
            for i in range(0, len(bn_titles), 50):
                batch = bn_titles[i : i + 50]
                resolved = get_english_titles_batch(batch)
                en_titles_map.update(resolved)
                progress.advance(task, advance=len(batch))
                time.sleep(0.1)  # Polite gap

        resolved_en_titles = list(en_titles_map.values())
        if not resolved_en_titles:
            ok(f"No English counterparts resolved for '{cat}'.")
            continue

        # Step 2: Batch filter to keep only Bangladesh-related English titles (50 at a time)
        related_en_titles = []
        with pipeline_progress() as progress:
            task = progress.add_task(f"Filtering {cat}", total=len(resolved_en_titles))
            for i in range(0, len(resolved_en_titles), 50):
                batch = resolved_en_titles[i : i + 50]
                matched = filter_english_pages_related_batch(batch)
                related_en_titles.extend(matched)
                progress.advance(task, advance=len(batch))
                time.sleep(0.1)  # Polite gap

        if not related_en_titles:
            ok(f"No Bangladesh-related English counterpart articles found for '{cat}'.")
            continue

        info(f"Fetching content for {len(related_en_titles)} matched articles...")

        # Step 3: Fetch content for only matched pages in parallel
        results = []
        with pipeline_progress() as progress:
            task = progress.add_task(f"Fetching {cat}", total=len(related_en_titles))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(get_english_content, title): title
                    for title in related_en_titles
                }
                for future in as_completed(futures):
                    title = futures[future]
                    text = future.result()
                    if text and len(text.strip()) > 100:
                        results.append((title, text))
                    progress.advance(task)

        # Write output (one article per line)
        cat_chunk_count = 0
        with (
            open(output_file, "w", encoding="utf-8") as f_out,
            open(output_titles_file, "w", encoding="utf-8") as f_titles,
        ):
            for en_title, text in results:
                f_titles.write(en_title + "\n")
                article_data = (
                    json.dumps({"text": f"{en_title}\n{text}"}, ensure_ascii=False) + "\n"
                )
                f_out.write(article_data)
                cat_chunk_count += 1

        ok(f"Wrote {cat_chunk_count} English chunks for '{cat}' to {output_file}")
        metrics_summary.append(
            f"  - {cat.capitalize()} : {len(results)} articles, {cat_chunk_count} chunks"
        )
        total_chunks += cat_chunk_count

    done_panel(
        "English download ready",
        metrics_summary + [f"Total chunks: {total_chunks}", f"Output Directory: {OUTPUT_DIR}"],
    )


if __name__ == "__main__":
    main()
