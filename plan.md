# Validation and Execution Plan: RTX 5060 (8GB VRAM) Pipeline

Use this guide to verify, execute, and evaluate the upgraded Bangla Hallucination Detection pipeline on your main machine equipped with an **RTX 5060 (8GB VRAM)**.

---

## Hardware Configuration (RTX 5060 8GB)
Your target machine has key capabilities that we leverage:
1.  **Native `bfloat16` Support:** The GPU supports fast, native bfloat16 calculations. Gemma 4 E4B will load and run at optimal performance.
2.  **Model Scaling:** `FacebookAI/xlm-roberta-large` (560M parameters) is fine-tuned with LoRA + cosine LR schedule + gradient clipping, fitting comfortably in VRAM.
3.  **VRAM Clearing:** An explicit GPU cache and IPC memory sweep executes between the Cross-Encoder and LLM Verifier phases to prevent VRAM allocation overflows.

---

## Step 1: Preprocess the Datasets
Ensure your workspace dependencies are synchronized, then run the preprocessor to normalize text inputs to NFC and strip out zero-width characters:
```bash
just sync
just preprocess
```
*Expected result:* Preprocessed CSV files are generated under `dataset/processed/train.csv` and `dataset/processed/test.csv`.

---

## Step 2: Build the Dense RAG Index (with Bengali Wikipedia)
Set up your Bengali corpus directory and compile the dense search index using the state-of-the-art multilingual `BAAI/bge-m3` embedding model:

1.  **Download and Chunk Bengali Wikipedia (Heavy Task - Run on Main Machine):**
    Run the following Python script on your main machine to fetch, clean, chunk (200 words with 50-word overlap), and write the entire Bengali Wikipedia to `corpus/wikipedia.jsonl`:
    ```python
    # run_download_corpus.py
    import json
    import os
    from datasets import load_dataset

    print("Fetching Bengali Wikipedia from HuggingFace...")
    dataset = load_dataset("wikipedia", "20231101.bn", split="train")

    os.makedirs("corpus", exist_ok=True)
    out_path = "corpus/wikipedia.jsonl"
    print(f"Processing and writing articles to {out_path}...")

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            text = row["text"]
            words = text.split()
            # Chunking parameters: 200 words per chunk, 50 words overlap
            chunk_size = 200
            overlap = 50
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk.strip()) > 50:
                    f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    print("Corpus construction successfully complete!")
    ```
    Execute this script:
    ```bash
    uv run python run_download_corpus.py
    ```

2.  **Build Dense Vector Index:**
    ```bash
    just build-index
    ```
    *Expected result:* BGE-M3 encodes all chunks into normalized dense vectors and saves the index to `indexes/dense_index.pkl`.

---

## Step 3: Run the Upgraded Training Loop
To train the XLM-R Large cross-encoder, build the dynamic exemplars, and fit the 6-feature meta-classifier ensembler, execute:
```bash
just train
```

### What happens under the hood during `just train`:
1.  **Dense Retrieval:** Any training example with a `[NULL]` context query is searched against your BGE-M3 dense index. Evidence is retrieved only if the similarity is $\ge 0.5$ (preventing noise poisoning).
2.  **XLM-R Large LoRA Training:** A 5-fold Stratified Cross-Validation loop runs, saving the best adapter weights for each fold in `models/xlmr/`.
3.  **VRAM Sweep:** The training loop releases XLM-R models and sweeps GPU/IPC cache to clear VRAM.
4.  **Exemplar Indexing:** Builds a dense index of the training set (`indexes/exemplar_index.pkl`) to support dynamic few-shot prompting.
5.  **LLM Verification & Cultural Band Classification:** 
    *   For each query, Gemma 4 classifies the prompt band into `C0` (Global), `C1` (Bangladeshi), or `C2` (Time-sensitive) via next-token logits.
    *   Retrieves the top-3 nearest training exemplars, excluding the current target query to prevent leakage.
    *   Gemma 4 evaluates the few-shot prompt to assign a Faithful (Class 1) probability score. Borderline cases trigger a CoT reasoning pass, which asks the model to explain and conclude with `verdict: Faithful` or `verdict: Hallucinated` (parsed via regex).
6.  **6-Feature RandomForest Meta-Classifier fitting:** A `RandomForestClassifier` fits on OOF features:
    `X = [p_xlmr, p_llm, has_context, is_c0, is_c1, is_c2]`
    learning context-aware and band-sensitive ensembling rules automatically — no manual grid search.

*Expected result:* Evaluated metrics, fold scores, and the trained ensembler are saved to `models/blender_config.pkl`.

---

## Step 4: Run Inference and Generate Submission
To make predictions on the test dataset and format your final submission:
```bash
just predict
```
*Expected result:* The script runs ensemble predictions across the 5 folds of XLM-R Large, queries Gemma 4 for few-shot verifier scores and cultural bands, blends them using the 6-feature meta-classifier, and outputs the final submission files to `submissions/submission.csv` and `submissions/submission_debug.csv`.
