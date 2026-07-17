import json
import os
import re
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


def classify_article(title, text):
    """Classifies if a Wikipedia article belongs to granular categories.

    Returns:
        A list of category strings.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Extract candidate category lines (short lines at the end without punctuation)
    category_lines = []
    for line in reversed(lines):
        if (
            len(line) < 100
            and not line.endswith("।")
            and not line.endswith(".")
            and not line.endswith("?")
        ):
            category_lines.append(line)
        else:
            if len(category_lines) >= 3:
                break

    categories = []

    # 1. Define category indicators for granular classification
    is_writer_poet = False
    is_politician_leader = False
    is_scientist_scholar = False
    is_artist_performer = False
    is_athlete = False
    is_other_person = False

    is_country_state = False
    is_city_town = False
    is_admin_unit = False
    is_village = False
    is_water_body = False
    is_mountain_island = False
    is_other_place = False

    is_war_battle = False
    is_hist_event = False

    is_book_lit = False
    is_movie_drama = False
    is_music_song = False
    is_art_fest = False

    is_edu_inst = False
    is_company_biz = False
    is_gov_body = False
    is_club_assoc = False

    is_rel_text = False
    is_deity = False
    is_rel_place = False

    is_animal = False
    is_plant = False

    is_math = False
    is_phys_chem = False
    is_astronomy = False
    is_tech = False
    is_med_bio = False

    # Check category suffixes
    for cat in category_lines:
        # People sub-types
        if any(cat.endswith(kw) for kw in ["কবি", "লেখক", "ঔপন্যাসিক", "প্রাবন্ধিক", "অনুবাদক"]):
            is_writer_poet = True
        elif any(
            cat.endswith(kw)
            for kw in [
                "রাজনীতিবিদ",
                "নেতা",
                "শাসক",
                "সম্রাট",
                "সুলতান",
                "খলিফা",
                "রাজা",
                "প্রেসিডেন্ট",
                "প্রধানমন্ত্রী",
                "মন্ত্রী",
            ]
        ):
            is_politician_leader = True
        elif any(
            cat.endswith(kw)
            for kw in ["বিজ্ঞানী", "গবেষক", "অধ্যাপক", "دار্শনিক", "সমাজবিজ্ঞানী", "ইতিহাসবিদ", "ভাষাবিদ"]
        ):
            is_scientist_scholar = True
        elif any(
            cat.endswith(kw)
            for kw in [
                "অভিনেতা",
                "অভিনেত্রী",
                "সঙ্গীতশিল্পী",
                "গায়ক",
                "চিত্রশিল্পী",
                "নর্তক",
                "সুরকার",
                "গীতিকার",
                "চলচ্চিত্র নির্মাতা",
            ]
        ):
            is_artist_performer = True
        elif any(cat.endswith(kw) for kw in ["খেলোয়াড়", "ক্রিকেটার", "ফুটবলার", "অ্যাথলেট", "দাবাড়ু"]):
            is_athlete = True
        elif any(kw in cat for kw in ["জন্ম", "মৃত্যু", "জীবনী"]) or any(
            cat.endswith(kw) for kw in ["ব্যক্তি", "ব্যক্তিত্ব", "মহিলা", "নারী", "পুরুষ"]
        ):
            is_other_person = True

        # Places sub-types
        if any(cat.endswith(kw) for kw in ["দেশ", "রাষ্ট্র", "রাজ্য", "প্রদেশ"]):
            is_country_state = True
        elif any(cat.endswith(kw) for kw in ["শহর", "নগর", "রাজধানী", "পৌরসভা"]):
            is_city_town = True
        elif any(
            cat.endswith(kw)
            for kw in ["বিভাগ", "জেলা", "উপজেলা", "ইউনিয়ন", "থানা", "ওয়ার্ড", "প্রশাসনিক এলাকা"]
        ):
            is_admin_unit = True
        elif any(cat.endswith(kw) for kw in ["গ্রাম"]):
            is_village = True
        elif any(
            cat.endswith(kw)
            for kw in ["নদী", "সাগর", "মহাসাগর", "হ্রদ", "খাল", "বিল", "ঝরনা", "উপসাগর", "জলাশয়"]
        ):
            is_water_body = True
        elif any(cat.endswith(kw) for kw in ["পাহাড়", "পর্বত", "দ্বীপ", "উপদ্বীপ", "গিরিপথ", "পর্বতমালা"]):
            is_mountain_island = True
        elif any(
            cat.endswith(kw) for kw in ["জনবহুল স্থান", "ভৌগোলিক স্থান", "ঐতিহাসিক স্থান", "স্থান", "অঞ্চল"]
        ):
            is_other_place = True

        # History checks
        if any(cat.endswith(kw) for kw in ["যুদ্ধ", "সামরিক অভিযান", "যুদ্ধবিগ্রহ"]):
            is_war_battle = True
        elif any(cat.endswith(kw) for kw in ["আন্দোলন", "বিপ্লব", "দাঙ্গা", "চুক্তি", "ঘটনা", "ইতিহাস"]):
            is_hist_event = True

        # Culture checks
        if any(cat.endswith(kw) for kw in ["উপন্যাস", "কবিতা", "গ্রন্থ", "সাহিত্য", "গল্প", "বই"]):
            is_book_lit = True
        elif any(cat.endswith(kw) for kw in ["চলচ্চিত্র", "নাটক", "থিয়েটার", "টেলিভিশন অনুষ্ঠান"]):
            is_movie_drama = True
        elif any(cat.endswith(kw) for kw in ["গান", "সঙ্গীত", "বাদ্যযন্ত্র", "অ্যালবাম"]):
            is_music_song = True
        elif any(cat.endswith(kw) for kw in ["চিত্রকর্ম", "ভাস্কর্য", "মেলা", "উৎসব", "ঐতিহ্য", "শিল্প"]):
            is_art_fest = True

        # Organizations checks
        if any(
            cat.endswith(kw) for kw in ["বিশ্ববিদ্যালয়", "কলেজ", "বিদ্যালয়", "একাডেমি", "শিক্ষা প্রতিষ্ঠান"]
        ):
            is_edu_inst = True
        elif any(
            cat.endswith(kw) for kw in ["কোম্পানি", "банк", "কারখানা", "কর্পোরেশন", "ব্যবসায়িক প্রতিষ্ঠান"]
        ):
            is_company_biz = True
        elif any(cat.endswith(kw) for kw in ["সংসদ", "মন্ত্রণালয়", "বিভাগ", "আদালত", "সরকার"]):
            is_gov_body = True
        elif any(cat.endswith(kw) for kw in ["ক্লাব", "দল", "সংগঠন", "সংস্থা", "সমিতি"]):
            is_club_assoc = True

        # Religion checks
        if any(cat.endswith(kw) for kw in ["মতবাদ", "ধর্মগ্রন্থ", "পুরাণ", "ধর্মবিশ্বাস"]):
            is_rel_text = True
        elif any(cat.endswith(kw) for kw in ["দেবতা", "দেবী", "পয়গম্বর", "অবতার", "ধর্মীয় ব্যক্তিত্ব"]):
            is_deity = True
        elif any(cat.endswith(kw) for kw in ["মন্দির", "মসজিদ", "গির্জা", "বিহার", "ধর্মীয় স্থান"]):
            is_rel_place = True

        # Nature checks
        if any(cat.endswith(kw) for kw in ["প্রাণী", "পাখি", "মাছ", "পতঙ্গ", "স্তন্যপায়ী", "সরীসৃপ"]):
            is_animal = True
        elif any(cat.endswith(kw) for kw in ["উদ্ভিদ", "ফুল", "গাছ", "ফল", "বৃক্ষ"]):
            is_plant = True

        # Science checks
        if any(cat.endswith(kw) for kw in ["গণিত", "জ্যামিতি", "বীজগণিত"]):
            is_math = True
        elif any(cat.endswith(kw) for kw in ["পদার্থবিজ্ঞান", "রসায়ন"]):
            is_phys_chem = True
        elif any(cat.endswith(kw) for kw in ["জ্যোতির্বিজ্ঞান", "মহাকাশ", "গ্রহ", "তারা", "নক্ষত্র"]):
            is_astronomy = True
        elif any(cat.endswith(kw) for kw in ["প্রযুক্তি", "কম্পিউটার", "সফটওয়্যার", "ইন্টারনেট", "মোবাইল"]):
            is_tech = True
        elif any(cat.endswith(kw) for kw in ["চিকিৎসা", "জীববিজ্ঞান", "রোগ", "ঔষধ"]):
            is_med_bio = True

    # Also check intro for person patterns if no person sub-type detected yet
    intro = text[:200]
    is_any_person = any(
        [
            is_writer_poet,
            is_politician_leader,
            is_scientist_scholar,
            is_artist_performer,
            is_athlete,
            is_other_person,
        ]
    )
    if not is_any_person:
        if (
            re.search(r"\(\s*[০-৯]{4}\s*[-–—]", intro)
            or re.search(r"[-–—]\s*[০-৯]{4}\s*\)", intro)
            or re.search(r"\(\s*\d{4}\s*[-–—]", intro)
            or re.search(r"[-–—]\s*\d{4}\s*\)", intro)
        ):
            is_other_person = True

    # Check intro for place patterns if no place sub-type detected yet
    is_any_place = any(
        [
            is_country_state,
            is_city_town,
            is_admin_unit,
            is_village,
            is_water_body,
            is_mountain_island,
            is_other_place,
        ]
    )
    if not is_any_place:
        if "অবস্থিত" in intro or "স্থানাঙ্ক" in intro:
            is_other_place = True

    # Map flags to categories
    if is_writer_poet:
        categories.append("writers_poets")
    if is_politician_leader:
        categories.append("politicians_leaders")
    if is_scientist_scholar:
        categories.append("scientists_scholars")
    if is_artist_performer:
        categories.append("artists_performers")
    if is_athlete:
        categories.append("athletes")
    if is_other_person:
        categories.append("other_people")

    if is_country_state:
        categories.append("countries_states")
    if is_city_town:
        categories.append("cities_towns")
    if is_admin_unit:
        categories.append("administrative_units")
    if is_village:
        categories.append("villages")
    if is_water_body:
        categories.append("water_bodies")
    if is_mountain_island:
        categories.append("mountains_islands")
    if is_other_place:
        categories.append("other_places")

    if is_war_battle:
        categories.append("wars_battles")
    if is_hist_event:
        categories.append("historical_events_revolutions")

    if is_book_lit:
        categories.append("books_literature")
    if is_movie_drama:
        categories.append("movies_drama")
    if is_music_song:
        categories.append("music_songs")
    if is_art_fest:
        categories.append("art_festivals")

    if is_edu_inst:
        categories.append("educational_institutions")
    if is_company_biz:
        categories.append("companies_businesses")
    if is_gov_body:
        categories.append("government_bodies")
    if is_club_assoc:
        categories.append("clubs_associations")

    if is_rel_text:
        categories.append("religious_texts_doctrines")
    if is_deity:
        categories.append("deities_figures")
    if is_rel_place:
        categories.append("religious_places")

    if is_animal:
        categories.append("animals")
    if is_plant:
        categories.append("plants")

    if is_math:
        categories.append("mathematics")
    if is_phys_chem:
        categories.append("physics_chemistry")
    if is_astronomy:
        categories.append("astronomy_space")
    if is_tech:
        categories.append("technology_computing")
    if is_med_bio:
        categories.append("medicine_biology")

    if not categories:
        categories.append("other")

    return categories


# Configuration Constants
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.bn"
SPLIT = "train"
OUTPUT_DIR = "generated/wiki"
CHUNK_WORDS = 220
OVERLAP_WORDS = 50
MAX_ARTICLES = 0  # Limit article count for debugging; use 0 for the full split.
FILTER = True

# Enabled categories for the output RAG corpus (can enable/disable by adding/removing from here)
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
    # "other",
]


def main():
    banner(
        "Download wiki corpus",
        f"{DATASET} / {CONFIG} → {OUTPUT_DIR}",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    info(f"Loading dataset split '{SPLIT}'...")
    dataset = load_dataset(DATASET, CONFIG, split=SPLIT)
    total = len(dataset) if MAX_ARTICLES <= 0 else min(len(dataset), MAX_ARTICLES)
    info(f"Will process up to {total} articles")

    processed_count = 0
    kept_articles = 0
    skipped_articles = 0
    chunk_count = 0

    category_names = [
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
    category_counts = {cat: 0 for cat in category_names}

    # Open title and jsonl files per category
    title_files = {}
    jsonl_files = {}
    for cat in category_names:
        title_files[cat] = open(
            os.path.join(OUTPUT_DIR, f"titles_{cat}.txt"), "w", encoding="utf-8"
        )
        jsonl_files[cat] = open(os.path.join(OUTPUT_DIR, f"{cat}.jsonl"), "w", encoding="utf-8")

    with pipeline_progress() as progress:
        task = progress.add_task("Chunking wiki", total=total)
        for row in dataset:
            if MAX_ARTICLES > 0 and processed_count >= MAX_ARTICLES:
                break

            row_dict: dict = row
            title = str(row_dict["title"])
            text = str(row_dict["text"])

            # Classify article into categories
            categories = classify_article(title, text)

            # Track titles per category
            for cat in categories:
                category_counts[cat] += 1
                title_files[cat].write(title + "\n")

            # Filter by enabled categories if FILTER is enabled
            if FILTER:
                matched_enabled_categories = [
                    cat for cat in categories if cat in ENABLED_CATEGORIES
                ]
                if not matched_enabled_categories:
                    skipped_articles += 1
                    processed_count += 1
                    progress.advance(task)
                    continue
                target_categories = matched_enabled_categories
            else:
                target_categories = categories

            kept_articles += 1
            article_data = json.dumps({"text": f"{title}\n{text}"}, ensure_ascii=False) + "\n"
            for cat in target_categories:
                jsonl_files[cat].write(article_data)
            chunk_count += 1

            processed_count += 1
            if processed_count % 100 == 0 or processed_count == total:
                progress.update(
                    task,
                    description=f"Processing wiki · {chunk_count} articles",
                )
            progress.advance(task)

    # Close all files
    for tf in title_files.values():
        tf.close()
    for jf in jsonl_files.values():
        jf.close()

    ok(f"Wrote {chunk_count} chunks from {kept_articles} articles (skipped {skipped_articles})")

    # Display count summary of all categories
    metrics_summary = [
        f"Processed: {processed_count}",
        f"Kept RAG articles: {kept_articles}",
        f"Skipped RAG articles: {skipped_articles}",
    ]
    for cat in category_names:
        enabled_str = "ENABLED" if cat in ENABLED_CATEGORIES else "DISABLED"
        metrics_summary.append(
            f"  - {cat.capitalize()} ({enabled_str}): {category_counts[cat]} articles"
        )

    done_panel(
        "Corpus ready",
        metrics_summary + [f"Total chunks: {chunk_count}", f"Output Directory: {OUTPUT_DIR}"],
    )


if __name__ == "__main__":
    main()
