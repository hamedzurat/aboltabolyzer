# Typed RAG Corpora

Each subdirectory is one retrieval source. Put `*.jsonl` files with one JSON object per line:

```jsonl
{"text": "..."}
{"passage": "..."}
```

## Sources

The five sources separate cases where retrieval means different things. Do not split further until the labeled audit set shows a repeated failure mode. If you do not have curated material yet, leave a source empty rather than filling it with noisy mixed documents — `just make-rag` skips empty folders and prediction records `index_missing:<source>`.

```text
corpus/wiki/        # general facts (Bengali Wikipedia)
corpus/famous_bn/   # Bangladesh history / literature / geography
corpus/idioms/      # ভাবার্থ
corpus/literal/     # শাব্দিক অর্থ
corpus/grammar/     # Bangla grammar
```

```text
indexes/<source>.pkl
```

| Source | Role |
| --- | --- |
| `wiki` | Broad factual lookup |
| `famous_bn` | High-value Bangladesh/literature facts that often get swapped |
| `idioms` | Figurative phrase meanings (ভাবার্থ) |
| `literal` | Literal/compositional meanings (শাব্দিক অর্থ) |
| `grammar` | Short Bangla grammar rules and examples |

Task → source mapping lives in `src/evidence_policy.py`. Config lists sources as:

```toml
[rag]
sources = ["wiki", "famous_bn", "idioms", "literal", "grammar"]
```

## Build

```bash
just download-corpus                 # → corpus/wiki/wiki_bn.jsonl
just download-corpus --max-articles 200
just make-rag                        # all non-empty sources
just make-rag --source wiki          # one source
just make-rag --source grammar
```

## Writing good JSONL lines

Use compact, retrieval-friendly notes: one fact, one meaning, or one small rule/example family per line. BGE retrieval works better when each line is small enough that the prompt terms and answer terms land in the same passage. A practical size is **40–120 words** per line. Avoid dumping long textbook chapters into one row.

### `corpus/wiki/wiki_bn.jsonl`

```jsonl
{"text": "বাংলাদেশের রাজধানী ঢাকা। ঢাকা বাংলাদেশের প্রশাসনিক, অর্থনৈতিক ও সাংস্কৃতিক কেন্দ্র।"}
{"text": "পদ্মা নদী বাংলাদেশের অন্যতম প্রধান নদী। গঙ্গা নদী বাংলাদেশে প্রবেশ করার পর পদ্মা নামে পরিচিত।"}
{"text": "সুন্দরবন বাংলাদেশ ও ভারতের পশ্চিমবঙ্গে অবস্থিত একটি বৃহৎ ম্যানগ্রোভ বন। রয়েল বেঙ্গল টাইগারের জন্য সুন্দরবন বিখ্যাত।"}
{"text": "জাতিসংঘ ১৯৪৫ সালে প্রতিষ্ঠিত হয়। এর সদর দপ্তর যুক্তরাষ্ট্রের নিউ ইয়র্ক শহরে অবস্থিত।"}
```

### `corpus/famous_bn/bangladesh_facts.jsonl`

```jsonl
{"text": "বঙ্গবন্ধু শেখ মুজিবুর রহমান বাংলাদেশের স্বাধীনতা আন্দোলনের প্রধান নেতা। তিনি ১৯২০ সালের ১৭ মার্চ টুঙ্গিপাড়ায় জন্মগ্রহণ করেন।"}
{"text": "বাংলাদেশের স্বাধীনতা দিবস ২৬ মার্চ। বিজয় দিবস ১৬ ডিসেম্বর। এই দুই জাতীয় দিবসের তারিখ আলাদা।"}
{"text": "অপারেশন সার্চলাইট ১৯৭১ সালের ২৫ মার্চ রাতে পাকিস্তানি সেনাবাহিনীর সামরিক অভিযান। মুজিবনগর সরকার গঠিত হয় ১৯৭১ সালের ১০ এপ্রিল।"}
{"text": "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালে জন্মগ্রহণ করেন এবং ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার লাভ করেন। কাজী নজরুল ইসলাম বাংলাদেশের জাতীয় কবি।"}
```

### `corpus/idioms/bangla_idioms.jsonl`

```jsonl
{"text": "অন্ধের যষ্টি: একমাত্র অবলম্বন বা ভরসা। কারও শেষ আশ্রয় বোঝাতে এই বাগধারা ব্যবহৃত হয়।"}
{"text": "গাছে কাঁঠাল গোঁফে তেল: কাজ সম্পন্ন হওয়ার আগেই লাভের আশা করা বা আগাম আনন্দ করা।"}
{"text": "লাঠালাঠি: ঝগড়া-বিবাদ বা মারামারি বোঝায়। প্রসঙ্গভেদে বিরোধপূর্ণ পরিস্থিতির ভাবার্থে ব্যবহৃত হয়।"}
{"text": "জো-হুকুমের দল: যারা নিজস্ব বিচার না করে শুধু আদেশ পালন করে; আজ্ঞাবহ লোকজন।"}
```

### `corpus/literal/literal_meanings.jsonl`

```jsonl
{"text": "ফ্ল্যাট শব্দের শাব্দিক অর্থ চ্যাপ্টা, সমতল বা সমান পৃষ্ঠবিশিষ্ট। বাসার অর্থে ফ্ল্যাট আলাদা ব্যবহার।"}
{"text": "জলহস্তী শব্দের শাব্দিক অর্থ জল বা পানির সঙ্গে সম্পর্কিত হস্তী; প্রচলিত অর্থে এটি হিপোপটেমাস প্রাণীকে বোঝায়।"}
{"text": "দূরবীন শব্দের শাব্দিক অর্থ দূরের বস্তু দেখার যন্ত্র। দূর + বীন বা দর্শন অর্থের সমন্বয়ে শব্দটি গঠিত।"}
{"text": "নববর্ষ শব্দের শাব্দিক অর্থ নতুন বছর। নব অর্থ নতুন এবং বর্ষ অর্থ বছর।"}
```

### Grammar

Include terms likely to appear in prompts: `সমাস`, `সন্ধি`, `কারক`, `বিভক্তি`, `ব্যাসবাক্য`, `ধাতু`, `ক্রিয়া পদ`, `কর্মধারয়`, `তৎপুরুষ`, `দ্বন্দ্ব`, `বহুব্রীহি`.

Example `corpus/grammar/somash.jsonl`:

```jsonl
{"text": "সমাস: দুই বা ততোধিক পদের মিলনে একটি পদ গঠিত হলে তাকে সমাস বলে। কর্মধারয় সমাসে বিশেষণ ও বিশেষ্য পদ মিলে এক অর্থ প্রকাশ করে; যেমন নীলকমল = নীল যে কমল।"}
{"text": "তৎপুরুষ সমাস: যে সমাসে পরপদের অর্থ প্রধান থাকে তাকে তৎপুরুষ সমাস বলে। উদাহরণ: রাজপুত্র = রাজার পুত্র, ঘরছাড়া = ঘর থেকে ছাড়া।"}
{"text": "দ্বন্দ্ব সমাস: যে সমাসে উভয় পদের অর্থ সমানভাবে প্রধান থাকে তাকে দ্বন্দ্ব সমাস বলে। উদাহরণ: মা-বাবা, দিনরাত, সুখদুঃখ।"}
{"text": "বহুব্রীহি সমাস: যে সমাসে সমস্যমান পদগুলোর কোনোটির অর্থ প্রধান নয়, অন্য কোনো ব্যক্তি বা বস্তুকে বোঝায়, তাকে বহুব্রীহি সমাস বলে। উদাহরণ: চন্দ্রমুখী = চন্দ্রের মতো মুখ যার।"}
```

Example `corpus/grammar/sandhi.jsonl`:

```jsonl
{"text": "সন্ধি: পাশাপাশি ধ্বনির মিলনে ধ্বনিগত পরিবর্তনকে সন্ধি বলে। স্বরসন্ধিতে দুই স্বরধ্বনির মিলন ঘটে; যেমন বিদ্যা + আলয় = বিদ্যালয়।"}
{"text": "ব্যঞ্জনসন্ধি: ব্যঞ্জনধ্বনির সঙ্গে ব্যঞ্জন বা স্বরধ্বনির মিলনে যে পরিবর্তন হয় তাকে ব্যঞ্জনসন্ধি বলে।"}
```

Example `corpus/grammar/karok.jsonl`:

```jsonl
{"text": "কারক: বাক্যে ক্রিয়ার সঙ্গে বিশেষ্য বা সর্বনামের সম্পর্ককে কারক বলে। কর্তা কারক ক্রিয়ার কর্তা বোঝায়; কর্ম কারক ক্রিয়ার কাজ যার উপর ঘটে তাকে বোঝায়।"}
{"text": "বিভক্তি: শব্দের সঙ্গে যুক্ত হয়ে বাক্যে শব্দের কারক বা সম্পর্ক প্রকাশ করে এমন চিহ্নকে বিভক্তি বলে। যেমন -কে, -র, -তে, -য়।"}
```

## Suggested starter files

```text
corpus/wiki/wiki_bn.jsonl
corpus/famous_bn/bangladesh_history.jsonl
corpus/famous_bn/bangla_literature.jsonl
corpus/idioms/bangla_idioms.jsonl
corpus/literal/literal_meanings.jsonl
corpus/grammar/somash.jsonl       # সমাস, ব্যাসবাক্য, examples
corpus/grammar/sandhi.jsonl       # স্বরসন্ধি / ব্যঞ্জনসন্ধি rules
corpus/grammar/karok.jsonl        # কারক, বিভক্তি
corpus/grammar/parts_of_speech.jsonl
```

## Legacy files

Migrate flat legacy files if needed:

```bash
mv corpus/facts.jsonl corpus/wiki/
mv corpus/bcs_facts.jsonl corpus/famous_bn/
```
