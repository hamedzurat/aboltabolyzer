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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config_utils import resolve_quantization_mode, resolve_section

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)

console = Console()


def _resolve_torch_dtype(dtype_name, default):
    if dtype_name in (None, "auto"):
        return default
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtype_map[str(dtype_name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype in config: {dtype_name}") from exc


class ExemplarRetriever:
    """Retrieves dynamic, leakage-free training exemplars for in-context learning (few-shot)."""

    def __init__(self, config):
        self.config = config
        self.rag_config = resolve_section(config, "rag")
        self.model_name = self.rag_config["model_name"]
        self.exemplar_path = self.rag_config["exemplar_index_path"]
        self.batch_size = self.rag_config.get("batch_size", 32)
        self.max_seq_length = self.rag_config.get("max_seq_length", None)
        self.model = None
        self.exemplars = []
        self.embeddings = None

    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            from src.config_utils import resolve_model_path

            resolved_path = resolve_model_path(self.model_name)
            self.model = SentenceTransformer(resolved_path)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length
            if self.model.device.type == "cuda":
                self.model = self.model.half()

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
            texts_to_encode,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=self.batch_size,
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
        self.model_loader = gemma_config.get("model_loader", "multimodal_lm")
        self.load_in = resolve_quantization_mode(gemma_config)
        self.device_map = gemma_config.get("device_map", "cuda:0")
        self.cuda_max_memory = gemma_config.get("cuda_max_memory")
        self.cpu_max_memory = gemma_config.get("cpu_max_memory", "32GiB")
        self.offload_folder = gemma_config.get("offload_folder", "models/offload/gemma")
        self.llm_int8_enable_fp32_cpu_offload = gemma_config.get(
            "llm_int8_enable_fp32_cpu_offload", False
        )
        self.torch_dtype_name = gemma_config.get("torch_dtype", "bfloat16")
        self.bnb_compute_dtype_name = gemma_config.get("bnb_4bit_compute_dtype", "bfloat16")
        self.clear_cuda_before_load = gemma_config.get("clear_cuda_before_load", True)
        self.max_input_tokens = gemma_config.get("max_input_tokens")
        self.exemplar_top_k = int(gemma_config.get("exemplar_top_k", 3))
        self.classify_cultural_band_enabled = gemma_config.get("classify_cultural_band", True)
        self.enable_think_pass = gemma_config.get("enable_think_pass", True)
        self.use_inputs_embeds_for_forward = gemma_config.get(
            "use_inputs_embeds_for_forward", False
        )
        self.use_language_model_direct = gemma_config.get("use_language_model_direct", False)
        self.conf_threshold = gemma_config["confidence_threshold"]
        self.disagree_threshold = gemma_config.get("disagree_threshold", 0.25)
        self.force_think_c0_null = gemma_config.get("force_think_c0_null", True)
        self.force_think_c2 = gemma_config.get("force_think_c2", True)
        self.max_think_tokens = gemma_config["max_think_tokens"]
        self.debug_log_path = "logs/debug_llm_verifier.jsonl"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_device = None
        self.model = None
        self.processor = None
        self.tokenizer = None

        self.token_f_ids = []
        self.token_h_ids = []

        self.exemplar_retriever = ExemplarRetriever(self.config)

    def load_model(self):
        from src.config_utils import resolve_model_path

        resolved_name = resolve_model_path(self.model_name)
        console.print(f"[bold cyan]Loading LLM verifier model:[/bold cyan] {self.model_name}")
        if self.clear_cuda_before_load and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        with Console().status("Initializing processor...", spinner="aesthetic"):
            if self.model_loader == "causal_lm":
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
                self.processor = self.tokenizer
            elif self.model_loader == "multimodal_lm":
                self.processor = AutoProcessor.from_pretrained(resolved_name)
                self.tokenizer = self.processor.tokenizer
            else:
                raise ValueError("gemma.model_loader must be 'causal_lm' or 'multimodal_lm'.")

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
        if self.load_in == "4bit" and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_resolve_torch_dtype(
                    self.bnb_compute_dtype_name, torch.bfloat16
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in == "8bit" and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
            )

        torch_dtype = _resolve_torch_dtype(
            self.torch_dtype_name,
            torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        load_kwargs = {
            "quantization_config": quant_config,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = self.device_map
            if self.device_map == "auto":
                max_memory = {"cpu": self.cpu_max_memory}
                if self.cuda_max_memory:
                    max_memory[0] = self.cuda_max_memory
                load_kwargs["max_memory"] = max_memory
                load_kwargs["offload_folder"] = self.offload_folder
        else:
            load_kwargs["device_map"] = None

        with Console().status(
            "Loading weights (this may take a few minutes)...", spinner="bouncingBar"
        ):
            model_class = (
                AutoModelForCausalLM
                if self.model_loader == "causal_lm"
                else AutoModelForMultimodalLM
            )
            self.model = model_class.from_pretrained(resolved_name, **load_kwargs)
            self.model.eval()
            self.input_device = self._infer_input_device()
            if self.use_language_model_direct and not (
                hasattr(self.model, "model")
                and hasattr(self.model.model, "language_model")
                and hasattr(self.model, "lm_head")
            ):
                raise ValueError(
                    "use_language_model_direct=true requires a Gemma-style multimodal "
                    "model with model.language_model and lm_head."
                )

        console.print("[green]✔ LLM verifier model loaded successfully![/green]")

    def _infer_input_device(self):
        if torch.cuda.is_available():
            for param in self.model.parameters():
                if param.device.type == "cuda":
                    return param.device
        return self.model.device

    def _prepare_inputs(self, prompt, use_inputs_embeds=None):
        tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
        original_truncation_side = getattr(tokenizer, "truncation_side", None)
        if tokenizer is not None and self.max_input_tokens is not None:
            tokenizer.truncation_side = "left"
        try:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
                truncation=self.max_input_tokens is not None,
                max_length=self.max_input_tokens,
            )
        finally:
            if tokenizer is not None and original_truncation_side is not None:
                tokenizer.truncation_side = original_truncation_side

        target_device = self.input_device or self.model.device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        if use_inputs_embeds is None:
            use_inputs_embeds = self.use_inputs_embeds_for_forward
        if use_inputs_embeds and not self.use_language_model_direct and "input_ids" in inputs:
            input_ids = inputs.pop("input_ids")
            with torch.inference_mode():
                inputs["inputs_embeds"] = self.model.get_input_embeddings()(input_ids)
        return inputs

    def _forward_logits(self, inputs):
        if not self.use_language_model_direct:
            try:
                outputs = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                outputs = self.model(**inputs)
            return outputs.logits

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        text_outputs = self.model.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = text_outputs.last_hidden_state[:, -1:, :]
        logits = self.model.lm_head(hidden_states)
        final_logit_softcapping = self.model.config.get_text_config().final_logit_softcapping
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        return logits

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

        inputs = self._prepare_inputs(prompt)

        with torch.inference_mode():
            logits = self._forward_logits(inputs)

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

    @staticmethod
    def _format_encoder_prior(p_xlmr, has_context, evidence):
        """Build a natural-language XLM-R prior block for Gemma prompts."""
        if p_xlmr is None:
            return ""

        p_xlmr = float(p_xlmr)
        if p_xlmr >= 0.65:
            verdict_hint = "সম্ভবত Faithful"
        elif p_xlmr <= 0.35:
            verdict_hint = "সম্ভবত Hallucinated"
        else:
            verdict_hint = "অনিশ্চিত"

        context_note = "প্রদত্ত প্রসঙ্গ আছে" if has_context else "প্রদত্ত প্রসঙ্গ নেই ([NULL])"
        evidence_note = (
            "প্রমাণ/তথ্যসূত্র উপলব্ধ"
            if str(evidence).strip() not in ("[NULL]", "", "None", "nan")
            else "কোনো প্রমাণ/তথ্যসূত্র নেই"
        )

        return (
            f"ক্রস-এনকোডার (XLM-R) ইঙ্গিত:\n"
            f"- P(Faithful) = {p_xlmr:.2f} → {verdict_hint}\n"
            f"- {context_note}; {evidence_note}\n"
            f"এই ইঙ্গিতটি সহায়ক; চূড়ান্ত বিচার তোমার।\n\n"
        )

    def _should_trigger_think(
        self,
        p_fast,
        p_xlmr,
        is_c0,
        is_c2,
        evidence,
        think_reasons,
    ):
        """Return True if any think trigger fires."""
        uncertainty = abs(p_fast - 0.5)
        if uncertainty < self.conf_threshold:
            think_reasons.append("uncertain_fast_pass")
            return True

        if p_xlmr is not None and abs(p_fast - float(p_xlmr)) > self.disagree_threshold:
            think_reasons.append("encoder_disagreement")
            return True

        evidence_is_null = str(evidence).strip() in ("[NULL]", "", "None", "nan")
        if self.force_think_c0_null and is_c0 >= 0.5 and evidence_is_null:
            think_reasons.append("c0_null_evidence")
            return True

        if self.force_think_c2 and is_c2 >= 0.5:
            think_reasons.append("c2_time_sensitive")
            return True

        return False

    def predict_single(
        self,
        evidence,
        prompt_bn,
        response_bn,
        p_xlmr=None,
        has_context=None,
        silent=True,
        write_log=True,
    ):
        if self.model is None:
            self.load_model()

        # 1. Classify the cultural band. On low-VRAM offload profiles this extra
        # forward pass is disabled because it is only used to route the think pass.
        if self.classify_cultural_band_enabled:
            is_c0, is_c1, is_c2 = self.predict_cultural_band(prompt_bn)
        else:
            is_c0, is_c1, is_c2 = 0.0, 1.0, 0.0

        # 2. Retrieve training exemplars for dynamic few-shot prompting
        if self.exemplar_top_k > 0:
            exemplars = self.exemplar_retriever.retrieve_exemplars(
                query=f"{prompt_bn} {response_bn}",
                exclude_prompt=prompt_bn,
                exclude_response=response_bn,
                top_k=self.exemplar_top_k,
            )
        else:
            exemplars = []

        # 3. Format prompt with exemplars and encoder prior
        encoder_prior = self._format_encoder_prior(p_xlmr, has_context, evidence)

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
            f"{encoder_prior}"
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

        inputs = self._prepare_inputs(prompt)

        with torch.inference_mode():
            logits = self._forward_logits(inputs)

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

        # Check think triggers (uncertainty, encoder disagreement, hard cases)
        think_reasons = []
        triggered_think = self._should_trigger_think(
            p_fast=p_llm,
            p_xlmr=p_xlmr,
            is_c0=is_c0,
            is_c2=is_c2,
            evidence=evidence,
            think_reasons=think_reasons,
        )
        p_llm_no_think = p_llm
        generated_text = ""

        if triggered_think and not self.enable_think_pass:
            think_reasons.append("think_pass_disabled")
            triggered_think = False

        if triggered_think:
            if not silent:
                reason_str = ", ".join(think_reasons) if think_reasons else "unknown"
                console.print(
                    f"[yellow]Think pass triggered ({reason_str}, p_fast={p_llm:.4f}).[/yellow]"
                )

            think_user_content = (
                f"{prompt_exemplars}"
                f"চলতি বিচার্য বিষয়:\n"
                f"{encoder_prior}"
                f"<evidence>\n{evidence}\n</evidence>\n"
                f"প্রশ্ন: {prompt_bn}\n"
                f"উত্তর: {response_bn}\n"
                f"ক্রস-এনকোডার ইঙ্গিত উপরে দেওয়া আছে; প্রয়োজন হলে তা বাতিল করো।\n"
                f"বিচার করো এবং নিজের ভাষায় ব্যাখ্যা কর। "
                f"শেষে অবশ্যই 'verdict: Faithful' অথবা 'verdict: Hallucinated' লিখবে।"
            )

            think_messages = [{"role": "user", "content": think_user_content}]
            think_prompt = self.processor.apply_chat_template(
                think_messages, tokenize=False, add_generation_prompt=True
            )

            think_inputs = self._prepare_inputs(think_prompt, use_inputs_embeds=False)

            with torch.inference_mode():
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
            "p_xlmr": None if p_xlmr is None else float(p_xlmr),
            "has_context": None if has_context is None else bool(has_context),
            "p_llm_no_think": float(p_llm_no_think),
            "triggered_think": bool(triggered_think),
            "think_reasons": think_reasons,
            "thinking_cot": generated_text_log,
            "p_llm_final": float(p_llm),
            "is_c0": float(is_c0),
            "is_c1": float(is_c1),
            "is_c2": float(is_c2),
        }
        if write_log:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            with open(self.debug_log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return p_llm, triggered_think

    def predict_dataset(self, df, p_xlmr=None, use_cache=True, debug_log_path=None):
        active_log_path = debug_log_path or self.debug_log_path
        original_log_path = self.debug_log_path
        self.debug_log_path = active_log_path

        cache = {}
        if use_cache and os.path.exists(active_log_path):
            try:
                with open(active_log_path, "r", encoding="utf-8") as lf:
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
                            )
                        except Exception:
                            continue
                if cache:
                    console.print(
                        f"[bold green]Loaded {len(cache)} existing predictions from debug log "
                        f"({active_log_path}).[/bold green]"
                    )
            except Exception as e:
                console.print(
                    f"[yellow]Could not read existing debug log: {e}. Starting fresh.[/yellow]"
                )

        if p_xlmr is None:
            if "p_xlmr" in df.columns:
                p_xlmr = df["p_xlmr"].values
            else:
                p_xlmr = [None] * len(df)

        if "has_context" in df.columns:
            has_context_values = df["has_context"].values
        else:
            has_context_values = (df["context"] != "[NULL]").values

        preds = []
        total_rows = len(df)
        think_count = 0

        try:
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
                    row_p_xlmr = p_xlmr[row_num - 1] if p_xlmr is not None else None
                    row_has_context = bool(has_context_values[row_num - 1])
                    key = (prompt, response)

                    if use_cache and key in cache:
                        prob, triggered_think = cache[key]
                    else:
                        if self.model is None:
                            self.load_model()
                        prob, triggered_think = self.predict_single(
                            evidence,
                            prompt,
                            response,
                            p_xlmr=row_p_xlmr,
                            has_context=row_has_context,
                            silent=True,
                            write_log=use_cache,
                        )

                    preds.append(prob)

                    if triggered_think:
                        think_count += 1

                    progress.update(
                        task,
                        description=f"Processed {row_num}/{total_rows} (Think triggers: {think_count})",
                    )
                    progress.advance(task)
        finally:
            self.debug_log_path = original_log_path

        console.print(
            f"[green]✔ Gemma predictions complete. Think triggered on "
            f"{think_count}/{total_rows} rows ({think_count / total_rows * 100:.1f}%).[/green]"
        )
        return np.array(preds)


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
    preds = verifier.predict_dataset(df_sample)
    console.print(f"Sample predictions: {preds}")


if __name__ == "__main__":
    main()
