# Aboltabolyzer Pipeline Examples

This file shows what happens to different input rows in the current pipeline: how the row is routed, whether RAG runs, which corpus is used, what prompt shape goes to the verifier, and how the final label is produced.

## Input Shape

Prediction starts from `generated/processed/test.csv`, produced by `just preprocess`.

Required columns:

```text
id, context, prompt_bn, response_bn
```

Preprocessing normalizes text, strips zero-width characters, collapses whitespace, and turns empty/null context into `[NULL]`.

The verifier outputs `p_llm`, then:

```text
label = 1 if p_llm >= decision.threshold else 0
```

Default threshold is `0.5`.

## High-Level Flow

```text
raw row
  -> preprocess text
  -> route task_type from context + prompt_bn + response_bn
  -> choose evidence policy
  -> maybe retrieve typed RAG evidence
  -> build verifier prompt
  -> fast F/H next-token score
  -> maybe think pass
  -> threshold to label
```

The important idea: `context` means "the evidence the verifier sees." If original context exists, it stays. If original context is `[NULL]` and RAG is allowed, retrieved text overwrites `context`.

## Router Rules

Routing is deterministic and lives in `src/router.py`.

If `context` is present:

| Input pattern                                    | `task_type`              |
| ------------------------------------------------ | ------------------------ |
| context present + famous entity + factual prompt | `famous_bn_fact_context` |
| context present + factual prompt                 | `context_grounded_fact`  |
| context present + other prompt                   | `context_grounded_other` |

If `context` is `[NULL]`:

| Prompt / response pattern                                                         | `task_type`                |
| --------------------------------------------------------------------------------- | -------------------------- |
| contains `ভাবার্থ`                                                                | `idiom_meaning_null`       |
| contains `শাব্দিক অর্থ`                                                           | `literal_meaning_null`     |
| work-rate keywords                                                                | `math_work_rate`           |
| speed/distance keywords                                                           | `math_speed_distance`      |
| profit/loss keywords                                                              | `math_profit_loss`         |
| average keywords                                                                  | `math_average`             |
| calendar/day keywords                                                             | `calendar_arithmetic`      |
| grammar keywords such as `সমাস`, `সন্ধি`, `কারক`, `বিভক্তি`, `ব্যাসবাক্য`, `ধাতু` | `bangla_grammar`           |
| translation keywords or Latin words                                               | `translation_or_bilingual` |
| famous Bangla/Bangladesh entity                                                   | `famous_bn_fact_null`      |
| factual question words such as `কোন`, `কে`, `কবে`, `কত`, `কোথায়`                 | `general_fact_null`        |
| none of the above                                                                 | `other_null`               |

## RAG Policy

RAG is skipped when original context exists. RAG is also skipped for math, calendar, and translation tasks.

| `task_type`                      | RAG? | Source                               |
| -------------------------------- | ---- | ------------------------------------ |
| `context_grounded_fact`          | no   | original context                     |
| `context_grounded_other`         | no   | original context                     |
| `famous_bn_fact_context`         | no   | original context                     |
| `general_fact_null`              | yes  | `corpus/wiki/`                       |
| factual `other_null`             | yes  | `corpus/wiki/`                       |
| non-factual `other_null`         | no   | none                                 |
| `famous_bn_fact_null`            | yes  | `corpus/famous_bn/`, fallback `wiki` |
| `idiom_meaning_null`             | yes  | `corpus/idioms/`                     |
| `literal_meaning_null`           | yes  | `corpus/literal/`                    |
| `bangla_grammar`                 | yes  | `corpus/grammar/`                    |
| `math_*` / `calendar_arithmetic` | no   | none                                 |
| `translation_or_bilingual`       | no   | none                                 |

If an index is missing, the row keeps `[NULL]` evidence and records:

```text
rag_skipped_reason = index_missing:<source>
```

## RAG Query

Default config:

```toml
[rag]
query_mode = "prompt"
top_k = 5
similarity_threshold = 0.5
max_evidence_tokens = 512
```

So the retrieval query is normally:

```text
prompt_bn
```

If `query_mode = "prompt_response"`, the query becomes:

```text
prompt_bn + " " + response_bn
```

Retrieved hits above `similarity_threshold` are joined into one evidence string and truncated to `max_evidence_tokens`.

## Verifier Prompt Shape

For each row, `GemmaVerifier.predict_single()` creates this user prompt content:

```text
Ex 1
E: ...optional few-shot evidence...
Q: ...
A: ...
V: F or H

Task: <task_type>
Rule: <task-specific instruction>
<evidence>
<current evidence: original context, RAG text, or [NULL]>
</evidence>
Q: <prompt_bn>
A: <response_bn>
Return one token only: F = faithful/correct/label 1; H = hallucinated/wrong/label 0.
V:
```

The fast pass does not generate a full answer. It looks at the next-token probabilities for F-like tokens and H-like tokens:

```text
p_fast = P(F) / (P(F) + P(H))
```

Initially:

```text
p_llm = p_fast
```

## Think Pass

Think pass may run if enabled and one of these triggers fires:

| Trigger                                                         | Reason                       |
| --------------------------------------------------------------- | ---------------------------- |
| `p_fast` between `think_conf_low` and `think_conf_high`         | `near_threshold`             |
| `famous_bn_fact_null`                                           | `famous_bn_fact_null`        |
| `context_grounded_fact` with multiple dates/entities in context | `multi_entity_context`       |
| math or calendar task                                           | `math_needs_check`           |
| idiom/literal with missing-looking evidence                     | `lexical_missing_evidence`   |
| RAG-eligible task but evidence lacks key prompt tokens          | `evidence_missing_keyphrase` |

Default 16GB and 8GB profiles enable the explicit think pass.

Think prompt shape:

```text
Task: <task_type>
Rule: <task-specific instruction>
<evidence>
...
</evidence>
Q: <prompt_bn>
A: <response_bn>
Write exactly this format, with the verdict first:
verdict: Faithful|Hallucinated
confidence: strong|likely|uncertain
reason: <one short English sentence>
```

Parsed think verdict maps to scores:

| Verdict      | Confidence | `p_llm` |
| ------------ | ---------- | ------: |
| Faithful     | strong     |    0.90 |
| Faithful     | likely     |    0.75 |
| Faithful     | uncertain  |    0.50 |
| Hallucinated | uncertain  |    0.50 |
| Hallucinated | likely     |    0.25 |
| Hallucinated | strong     |    0.10 |

If parsing fails, the pipeline keeps `p_fast`.

## Examples

### 1. Context-Grounded Factual Row

Input:

```csv
id,context,prompt_bn,response_bn
1,"রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার পান।","রবীন্দ্রনাথ কত সালে নোবেল পুরস্কার পান?","১৯১৩ সালে"
```

Routing:

```text
context present -> factual prompt contains "কত সালে"
task_type = context_grounded_fact
```

Evidence:

```text
RAG skipped: original_context_present
evidence = original context
```

Verifier instruction:

```text
Judge strictly against the provided context. If the answer contradicts the context,
uses a nearby wrong date/person/place, or omits a required qualifier, mark Hallucinated.
```

Expected behavior:

```text
p_fast should be high if the model sees answer matches context.
label likely 1.
```

### 2. Context-Grounded Contradiction

Input:

```csv
id,context,prompt_bn,response_bn
2,"রবীন্দ্রনাথ ঠাকুর ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার পান।","রবীন্দ্রনাথ কত সালে নোবেল পুরস্কার পান?","১৯১২ সালে"
```

Routing and evidence are the same as example 1.

Prompt difference:

```text
Question: রবীন্দ্রনাথ কত সালে নোবেল পুরস্কার পান?
Answer: ১৯১২ সালে
```

Expected behavior:

```text
The answer contradicts the evidence, so p_llm should go below threshold.
label likely 0.
```

### 3. No-Context General Fact

Input:

```csv
id,context,prompt_bn,response_bn
3,[NULL],"বাংলাদেশের রাজধানী কোনটি?","ঢাকা"
```

Routing:

```text
context is [NULL]
prompt contains factual word "কোন"
task_type = general_fact_null
```

Evidence:

```text
RAG source = wiki
query = "বাংলাদেশের রাজধানী কোনটি?"
index = indexes/wiki.pkl
evidence_source = rag:wiki
```

Example retrieved evidence:

```text
বাংলাদেশের রাজধানী ঢাকা। ঢাকা বাংলাদেশের প্রশাসনিক, অর্থনৈতিক ও সাংস্কৃতিক কেন্দ্র।
```

Verifier instruction:

```text
Check the factual answer carefully. Watch for swapped people, dates, places,
nearby events, total-vs-part numbers, and famous Bangladesh/literature confusions.
```

Expected behavior:

```text
label likely 1.
```

### 4. Famous Bangladesh/Literature Fact

Input:

```csv
id,context,prompt_bn,response_bn
4,[NULL],"রবীন্দ্রনাথ কত সালে জন্মগ্রহণ করেন?","১৮৬১ সালে"
```

Routing:

```text
context is [NULL]
prompt contains famous entity "রবীন্দ্রনাথ"
task_type = famous_bn_fact_null
```

Evidence:

```text
preferred source = famous_bn
fallback source = wiki
query = "রবীন্দ্রনাথ কত সালে জন্মগ্রহণ করেন?"
```

Think pass:

```text
famous_bn_fact_null always triggers think if enable_think_pass = true.
```

Expected behavior:

```text
Fast pass scores F/H.
Think pass may override with 0.90/0.75/0.25/0.10 style score.
label likely 1 if evidence/model confirms ১৮৬১.
```

### 5. Idiom Meaning

Input:

```csv
id,context,prompt_bn,response_bn
5,[NULL],"‘জো-হুকুমের দল’ এর ভাবার্থ কী?","আজ্ঞাবহ লোকজন"
```

Routing:

```text
context is [NULL]
prompt contains "ভাবার্থ"
task_type = idiom_meaning_null
```

Evidence:

```text
RAG source = idioms
query = "‘জো-হুকুমের দল’ এর ভাবার্থ কী?"
```

Example corpus line:

```jsonl
{
  "text": "জো-হুকুমের দল: যারা নিজস্ব বিচার না করে শুধু আদেশ পালন করে; আজ্ঞাবহ লোকজন।"
}
```

Verifier instruction:

```text
The question asks for Bengali ভাবার্থ. Use Bengali idiom/phrase knowledge.
Absence from retrieved evidence is not enough to mark hallucinated.
Judge whether the response is the correct figurative meaning.
```

Expected behavior:

```text
If evidence is good, label likely 1.
If idiom index is missing, evidence remains [NULL], but the instruction says not to reject only because evidence is missing.
```

### 6. Literal Meaning

Input:

```csv
id,context,prompt_bn,response_bn
6,[NULL],"‘ফ্ল্যাট’ এর শাব্দিক অর্থ কী?","চ্যাপ্টা"
```

Routing:

```text
context is [NULL]
prompt contains "শাব্দিক অর্থ"
task_type = literal_meaning_null
```

Evidence:

```text
RAG source = literal
query = "‘ফ্ল্যাট’ এর শাব্দিক অর্থ কী?"
```

Verifier instruction:

```text
The question asks for শাব্দিক অর্থ. Judge the literal/compositional meaning.
Do not reject only because there is no external evidence.
```

Expected behavior:

```text
label likely 1 if "চ্যাপ্টা" is accepted as literal meaning.
```

### 7. Bangla Grammar

Input:

```csv
id,context,prompt_bn,response_bn
7,[NULL],"‘নীলকমল’ শব্দটির সমাস কী?","কর্মধারয় সমাস"
```

Routing:

```text
context is [NULL]
prompt contains "সমাস"
task_type = bangla_grammar
```

Evidence:

```text
RAG source = grammar
query = "‘নীলকমল’ শব্দটির সমাস কী?"
```

Example corpus line:

```jsonl
{
  "text": "সমাস: দুই বা ততোধিক পদের মিলনে একটি পদ গঠিত হলে তাকে সমাস বলে। কর্মধারয় সমাসে বিশেষণ ও বিশেষ্য পদ মিলে এক অর্থ প্রকাশ করে; যেমন নীলকমল = নীল যে কমল।"
}
```

Verifier instruction:

```text
Judge by Bangla grammar rules. Use evidence if helpful, but missing evidence
alone is not H. Accept minor spelling variants when the category is clear.
```

Expected behavior:

```text
label likely 1.
```

### 8. Math

Input:

```csv
id,context,prompt_bn,response_bn
8,[NULL],"একটি গাড়ির গতিবেগ ঘণ্টায় ৬০ কিমি হলে ২ ঘণ্টায় দূরত্ব কত?","১২০ কিমি"
```

Routing:

```text
context is [NULL]
prompt contains speed/distance keywords
task_type = math_speed_distance
```

Evidence:

```text
RAG skipped: task_policy:math_speed_distance
evidence = [NULL]
```

Verifier instruction:

```text
Calculate the answer step by step internally. Compare the calculated answer
with the response. Mark Faithful only if they match.
```

Think pass:

```text
math_* tasks trigger think if enable_think_pass = true.
```

Expected behavior:

```text
60 * 2 = 120, so label likely 1.
```

### 9. Translation or Bilingual

Input:

```csv
id,context,prompt_bn,response_bn
9,[NULL],"Translate 'river' into Bengali","নদী"
```

Routing:

```text
context is [NULL]
prompt contains English/Latin text and translation cue
task_type = translation_or_bilingual
```

Evidence:

```text
RAG skipped: task_policy:translation_or_bilingual
evidence = [NULL]
```

Verifier instruction:

```text
Judge whether the response is the correct English/Bengali translation or terminology.
Watch for antonyms, wrong technical terms, and person/title confusion.
```

Expected behavior:

```text
label likely 1.
```

### 10. Other Null

Input:

```csv
id,context,prompt_bn,response_bn
10,[NULL],"একটি সুন্দর বাংলা বাক্য লিখুন","বাংলাদেশ একটি সুন্দর দেশ।"
```

Routing:

```text
context is [NULL]
no factual/grammar/math/translation/famous/idiom/literal pattern
task_type = other_null
```

Evidence:

```text
should_use_rag checks whether the prompt is factual.
This prompt is not factual.
RAG skipped: other_null_not_factual
evidence = [NULL]
```

Verifier instruction:

```text
No context is provided. Judge using general knowledge only when confident.
If the answer is unsupported and not clearly correct, mark Hallucinated.
```

Expected behavior:

```text
This category is intentionally broad and weaker than the typed categories.
Inspect debug output for these rows carefully.
```

## Router Disabled Mode

If config has:

```toml
[router]
enabled = false
```

Then the pipeline bypasses typed routing and typed RAG:

```text
context present -> task_type = context_grounded_other
context [NULL]  -> task_type = other_null
rag_used = false
rag_skipped_reason = router_disabled
```

This is useful as a baseline mode, but it loses most of the task-specific behavior.

## Debug Columns to Inspect

After prediction, inspect:

```text
submissions/latest/submission_debug.csv
```

Useful columns:

| Column               | Meaning                                                   |
| -------------------- | --------------------------------------------------------- |
| `task_type`          | Router decision                                           |
| `rag_used`           | Whether RAG filled evidence                               |
| `rag_source`         | Which source was used                                     |
| `rag_skipped_reason` | Why retrieval did not happen                              |
| `evidence_source`    | `original_context`, `rag:<source>`, or `none`             |
| `evidence_relevance` | `provided`, `retrieved`, `retrieval_empty`, `no_evidence` |
| `n_retrieved`        | Number of passages above threshold                        |
| `retrieval_sim_max`  | Best retrieval similarity                                 |
| `p_fast`             | Fast F/H probability                                      |
| `triggered_think`    | Whether think pass ran                                    |
| `think_reasons`      | Why think pass ran                                        |
| `p_think`            | Think-pass mapped score, if any                           |
| `p_llm`              | Final score after fast/think logic                        |
| `label`              | Final thresholded label                                   |

## Reading One Row End to End

For any row, read the debug CSV like this:

1. Check `task_type`.
2. Check `rag_used`, `rag_source`, and `rag_skipped_reason`.
3. Read `context_original` and `context`.
4. If `context != context_original`, RAG replaced the evidence.
5. Check `p_fast`.
6. If `triggered_think = true`, check `think_reasons`, `verdict_parsed`, and `confidence_parsed`.
7. Check `p_llm` and compare it to `threshold`.

That tells you exactly why the row landed on label `0` or `1`.
