import json
import logging
import os
import pickle
import re
import tomllib

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub.utils import disable_progress_bars
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from src.config_utils import resolve_section

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)

console = Console()


class ExemplarRetriever:
    """Retrieves dynamic, leakage-free training exemplars for in-context learning (few-shot)."""

    def __init__(self, config):
        self.config = config
        self.model_name = config["rag"]["model_name"]
        self.exemplar_path = config["rag"]["exemplar_index_path"]
        self.model = None
        self.exemplars = []
        self.embeddings = None

    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            from src.config_utils import resolve_model_path

            resolved_path = resolve_model_path(self.model_name)
            # Dynamically select device based on hardware profile and free VRAM
            device = "cpu"
            profile = self.config.get("runtime", {}).get("hardware_profile", "16gb")
            if profile != "8gb" and torch.cuda.is_available():
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    # Need at least 6.0 GB of free memory to comfortably hold both BGE-M3 and Gemma-4-E4B-it
                    if free_mem >= 6 * 1024 * 1024 * 1024:
                        device = "cuda"
                        print("[ExemplarRetriever] Ample VRAM detected. Loading BGE-M3 on GPU.")
                    else:
                        print(f"[ExemplarRetriever] Limited VRAM ({free_mem / (1024**3):.2f} GB free). Loading BGE-M3 on CPU to reserve space for LLM.")
                except Exception:
                    pass
            else:
                print("[ExemplarRetriever] 8GB profile active. Loading BGE-M3 on CPU to reserve GPU space for LLM.")
            
            self.model = SentenceTransformer(resolved_path, device=device)

    def build_index(self, df):
        """Encodes and saves the training dataframe rows as exemplars."""
        self.load_model()

        self.exemplars = []
        texts_to_encode = []
        for idx, row in df.iterrows():
            exemplar_info = {
                "context": str(row["context"]),
                "prompt_bn": str(row["prompt_bn"]),
                "response_bn": str(row["response_bn"]),
                "label": int(row["label"]),
            }
            self.exemplars.append(exemplar_info)
            # The representation to match on is prompt + response
            texts_to_encode.append(f"{row['prompt_bn']} {row['response_bn']}")

        self.embeddings = self.model.encode(
            texts_to_encode, show_progress_bar=False, normalize_embeddings=True
        )

        os.makedirs(os.path.dirname(self.exemplar_path), exist_ok=True)
        with open(self.exemplar_path, "wb") as f:
            pickle.dump({"exemplars": self.exemplars, "embeddings": self.embeddings}, f)
        print(f"Saved {len(self.exemplars)} exemplars to dense index at {self.exemplar_path}")

    def load_index(self):
        if not os.path.exists(self.exemplar_path):
            return False
        with open(self.exemplar_path, "rb") as f:
            data = pickle.load(f)
            self.exemplars = data["exemplars"]
            self.embeddings = data["embeddings"]
        return True

    def retrieve_exemplars(self, query, exclude_prompt=None, exclude_response=None, top_k=3):
        """Retrieves top_k nearest exemplars, avoiding target leakage by filtering out exact matching inputs."""
        if self.embeddings is None:
            if not self.load_index():
                return []

        self.load_model()

        query_emb = self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[
            0
        ]
        similarities = np.dot(self.embeddings, query_emb)

        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            ex = self.exemplars[idx]
            # Leakage check: skip if it's the exact same query
            if exclude_prompt is not None and ex["prompt_bn"].strip() == exclude_prompt.strip():
                continue
            if (
                exclude_response is not None
                and ex["response_bn"].strip() == exclude_response.strip()
            ):
                continue
            results.append(ex)
            if len(results) >= top_k:
                break
        return results


class GemmaVerifier:
    def __init__(self, config_path="configs/config.toml"):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        gemma_config = resolve_section(self.config, "gemma")

        self.model_name = gemma_config["model_name"]
        self.load_in_4bit = gemma_config["load_in_4bit"]
        self.conf_threshold = gemma_config["confidence_threshold"]
        self.max_think_tokens = gemma_config["max_think_tokens"]
        self.debug_log_path = "logs/debug_llm_verifier.jsonl"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.tokenizer = None

        self.token_f_ids = []
        self.token_h_ids = []

        self.exemplar_retriever = ExemplarRetriever(self.config)

    def load_model(self):
        from src.config_utils import resolve_model_path

        resolved_name = resolve_model_path(self.model_name)
        console.print(f"[bold cyan]Loading Gemma 4 verifier model:[/bold cyan] {self.model_name}")

        with Console().status("Initializing processor...", spinner="aesthetic"):
            self.processor = AutoProcessor.from_pretrained(resolved_name)
            self.tokenizer = self.processor.tokenizer

            f_variants = ["F", " F", "faithful", " Faithful"]
            h_variants = ["H", " H", "hallucinated", " Hallucinated"]

            f_ids = []
            for t in f_variants:
                ids = self.tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    f_ids.append(ids[-1])

            h_ids = []
            for t in h_variants:
                ids = self.tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    h_ids.append(ids[-1])

            self.token_f_ids = list(set(f_ids))
            self.token_h_ids = list(set(h_ids))

        quant_config = None
        if self.load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        with Console().status(
            "Loading weights (this may take a few minutes)...", spinner="bouncingBar"
        ):
            self.model = AutoModelForMultimodalLM.from_pretrained(
                resolved_name,
                quantization_config=quant_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.model.eval()

        console.print("[green]✔ Gemma 4 model loaded successfully![/green]")

    def predict_cultural_band(self, prompt_bn):
        """Classifies the prompt into C0, C1, or C2 based on next-token logits."""
        if self.model is None:
            self.load_model()

        # Prompt asking the model to classify the cultural band of the query
        user_content = (
            f"নিচের প্রশ্নটি মনোযোগ দিয়ে পড়ো এবং এটি কোন ক্যাটাগরির তা নির্ধারণ করো:\n"
            f"প্রশ্ন: {prompt_bn}\n\n"
            f"ক্যাটাগরি সমূহ:\n"
            f"C0 - বিশ্বজনীন বা বৈজ্ঞানিক তথ্য (যেমন: গণিত, সাধারণ বিজ্ঞান, বিশ্ব ভূগোল)\n"
            f"C1 - বাংলাদেশ বা বাঙালি সংস্কৃতি-সংক্রান্ত তথ্য (যেমন: বাংলাদেশ ইতিহাস, বাংলা সাহিত্য, স্থানীয় সংস্কৃতি)\n"
            f"C2 - সাম্প্রতিক, পরিবর্তনশীল বা বিতর্কিত তথ্য (যেমন: সাম্প্রতিক খবর, সময়-সংবেদনশীল তথ্য)\n\n"
            f"ক্যাটাগরি (C0/C1/C2):"
        )

        messages = [{"role": "user", "content": user_content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += "C"  # Pre-fill assistant response with "C" so the next token is exactly "0", "1", or "2"!

        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        # Retrieve probabilities for the digit tokens "0", "1", "2"
        token_0_ids = [
            self.tokenizer.encode("0", add_special_tokens=False)[-1],
            self.tokenizer.encode(" 0", add_special_tokens=False)[-1],
        ]
        token_1_ids = [
            self.tokenizer.encode("1", add_special_tokens=False)[-1],
            self.tokenizer.encode(" 1", add_special_tokens=False)[-1],
        ]
        token_2_ids = [
            self.tokenizer.encode("2", add_special_tokens=False)[-1],
            self.tokenizer.encode(" 2", add_special_tokens=False)[-1],
        ]

        prob_c0 = sum(probs[tid].item() for tid in token_0_ids if tid is not None)
        prob_c1 = sum(probs[tid].item() for tid in token_1_ids if tid is not None)
        prob_c2 = sum(probs[tid].item() for tid in token_2_ids if tid is not None)

        sum_prob = prob_c0 + prob_c1 + prob_c2
        if sum_prob > 0:
            p_c0 = prob_c0 / sum_prob
            p_c1 = prob_c1 / sum_prob
            p_c2 = prob_c2 / sum_prob
        else:
            p_c0, p_c1, p_c2 = 0.33, 0.33, 0.34

        # Return the one-hot band features
        probs_list = [p_c0, p_c1, p_c2]
        pred_idx = np.argmax(probs_list)

        is_c0 = 1.0 if pred_idx == 0 else 0.0
        is_c1 = 1.0 if pred_idx == 1 else 0.0
        is_c2 = 1.0 if pred_idx == 2 else 0.0

        return is_c0, is_c1, is_c2

    def predict_single(self, evidence, prompt_bn, response_bn, silent=True):
        if self.model is None:
            self.load_model()

        # 1. Classify the cultural band
        is_c0, is_c1, is_c2 = self.predict_cultural_band(prompt_bn)

        # 2. Retrieve training exemplars for dynamic few-shot prompting
        exemplars = self.exemplar_retriever.retrieve_exemplars(
            query=f"{prompt_bn} {response_bn}",
            exclude_prompt=prompt_bn,
            exclude_response=response_bn,
            top_k=3,
        )

        # 3. Format prompt with exemplars
        prompt_exemplars = ""
        for idx, ex in enumerate(exemplars):
            ex_label_str = "F" if ex["label"] == 1 else "H"
            prompt_exemplars += (
                f"উদাহরণ {idx + 1}:\n"
                f"<evidence>\n{ex['context']}\n</evidence>\n"
                f"প্রশ্ন: {ex['prompt_bn']}\n"
                f"উত্তর: {ex['response_bn']}\n"
                f"বিচার (F/H):{ex_label_str}\n\n"
            )

        user_content = (
            f"{prompt_exemplars}"
            f"চলতি বিচার্য বিষয়:\n"
            f"<evidence>\n{evidence}\n</evidence>\n"
            f"প্রশ্ন: {prompt_bn}\n"
            f"উত্তর: {response_bn}"
        )

        messages = [{"role": "user", "content": user_content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Append target prefix completion so the model continues directly after the prompt
        prompt += "বিচার (F/H):"

        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        prob_f = sum(probs[tid].item() for tid in self.token_f_ids if tid is not None)
        prob_h = sum(probs[tid].item() for tid in self.token_h_ids if tid is not None)

        sum_prob = prob_f + prob_h
        if sum_prob > 0:
            p_f = prob_f / sum_prob
        else:
            p_f = 0.5

        p_llm = p_f

        # Check Confidence Gate
        uncertainty = abs(p_llm - 0.5)
        triggered_think = False
        p_llm_no_think = p_llm
        generated_text = ""  # initialised here so the log block is always safe

        if uncertainty < self.conf_threshold:
            triggered_think = True
            if not silent:
                console.print(
                    f"[yellow]Uncertain prediction (p_llm={p_llm:.4f}). Triggering thinking pass...[/yellow]"
                )

            think_user_content = (
                f"{prompt_exemplars}"
                f"চলতি বিচার্য বিষয়:\n"
                f"<evidence>\n{evidence}\n</evidence>\n"
                f"প্রশ্ন: {prompt_bn}\n"
                f"উত্তর: {response_bn}\n"
                f"বিচার করো এবং নিজের ভাষায় ব্যাখ্যা কর। শেষে অবশ্যই 'verdict: Faithful' অথবা 'verdict: Hallucinated' লিখবে।"
            )

            think_messages = [{"role": "user", "content": think_user_content}]
            think_prompt = self.processor.apply_chat_template(
                think_messages, tokenize=False, add_generation_prompt=True
            )

            think_inputs = self.processor(text=think_prompt, return_tensors="pt")
            think_inputs = {k: v.to(self.model.device) for k, v in think_inputs.items()}

            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **think_inputs,
                    max_new_tokens=self.max_think_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            input_len = think_inputs["input_ids"].shape[1]
            generated_tokens = gen_outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Parse the full-word verdict from the CoT output.
            # We instruct the model to write "verdict: Faithful" or "verdict: Hallucinated",
            # which are far less likely to appear accidentally inside Bengali prose than
            # single characters F / H.
            verdict_match = re.search(
                r"verdict\s*:\s*(Faithful|Hallucinated)",
                generated_text,
                re.IGNORECASE,
            )
            if verdict_match:
                p_llm = 1.0 if verdict_match.group(1).lower() == "faithful" else 0.0

        # Log debug data
        if triggered_think:
            generated_text_log = generated_text
        else:
            generated_text_log = None

        log_entry = {
            "evidence": evidence,
            "prompt": prompt_bn,
            "response": response_bn,
            "p_llm_no_think": float(p_llm_no_think),
            "triggered_think": bool(triggered_think),
            "thinking_cot": generated_text_log,
            "p_llm_final": float(p_llm),
            "is_c0": float(is_c0),
            "is_c1": float(is_c1),
            "is_c2": float(is_c2),
        }
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        with open(self.debug_log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return p_llm, triggered_think, is_c0, is_c1, is_c2

    def predict_dataset(self, df):
        cache = {}
        if os.path.exists(self.debug_log_path):
            try:
                with open(self.debug_log_path, "r", encoding="utf-8") as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            key = (entry["prompt"], entry["response"])
                            cache[key] = (
                                entry["p_llm_final"],
                                entry["triggered_think"],
                                entry.get("is_c0", 0.0),
                                entry.get("is_c1", 0.0),
                                entry.get("is_c2", 0.0),
                            )
                        except Exception:
                            continue
                if cache:
                    console.print(
                        f"[bold green]Loaded {len(cache)} existing predictions from debug log ({self.debug_log_path}).[/bold green]"
                    )
            except Exception as e:
                console.print(
                    f"[yellow]Could not read existing debug log: {e}. Starting fresh.[/yellow]"
                )

        preds = []
        c0_features = []
        c1_features = []
        c2_features = []
        total_rows = len(df)
        think_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(description="Running Gemma Verifier...", total=total_rows)

            for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
                evidence = str(row["context"])
                prompt = str(row["prompt_bn"])
                response = str(row["response_bn"])
                key = (prompt, response)

                if key in cache:
                    prob, triggered_think, is_c0, is_c1, is_c2 = cache[key]
                else:
                    if self.model is None:
                        self.load_model()
                    prob, triggered_think, is_c0, is_c1, is_c2 = self.predict_single(
                        evidence, prompt, response, silent=True
                    )

                preds.append(prob)
                c0_features.append(is_c0)
                c1_features.append(is_c1)
                c2_features.append(is_c2)

                if triggered_think:
                    think_count += 1

                progress.update(
                    task,
                    description=f"Processed {row_num}/{total_rows} (Gate triggers: {think_count})",
                )
                progress.advance(task)

        console.print(
            f"[green]✔ Gemma predictions complete. Gate triggered on {think_count}/{total_rows} rows ({think_count / total_rows * 100:.1f}%).[/green]"
        )
        return (
            np.array(preds),
            np.array(c0_features),
            np.array(c1_features),
            np.array(c2_features),
        )


def main():
    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)

    train_path = os.path.join(config["data"]["processed_dir"], "train.csv")
    if not os.path.exists(train_path):
        console.print(
            "[bold red]Preprocessed train data not found. Please run preprocess.py first.[/bold red]"
        )
        return

    df = pd.read_csv(train_path)
    df_sample = df.head(3)

    verifier = GemmaVerifier()
    console.print(
        Panel("[bold yellow]🤖 Gemma Verifier Test Phase[/bold yellow]", border_style="yellow")
    )
    preds, c0, c1, c2 = verifier.predict_dataset(df_sample)
    console.print(f"Sample predictions: {preds}")
    console.print(f"Sample categories: C0={c0}, C1={c1}, C2={c2}")


if __name__ == "__main__":
    main()
