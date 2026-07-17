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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config_utils import resolve_quantization_mode, resolve_section
from src.evidence_policy import map_think_verdict, should_trigger_think, task_instruction
from src.router import route_row
from src.tui import banner, console, info, ok, pipeline_progress, warn

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)

CACHE_VERSION = "verifier-cache-v2"
CACHE_METADATA_FIELDS = (
    "model_name",
    "model_loader",
    "load_in",
    "max_input_tokens",
    "enable_think_pass",
    "exemplar_top_k",
    "think_conf_low",
    "think_conf_high",
    "chat_template_enable_thinking_fast",
    "chat_template_enable_thinking_think",
)


def verifier_case_key(
    *,
    evidence,
    prompt,
    response,
    task_type,
    context_original,
    metadata=None,
):
    """Stable identity for one verifier inference case."""
    key = (
        CACHE_VERSION,
        str(task_type),
        str(context_original),
        str(evidence),
        str(prompt),
        str(response),
    )
    if metadata:
        key += tuple((field, str(metadata.get(field, ""))) for field in CACHE_METADATA_FIELDS)
    return key


def verifier_log_matches_metadata(entry, expected_metadata):
    if entry.get("cache_version") != CACHE_VERSION:
        return False
    for field in CACHE_METADATA_FIELDS:
        if str(entry.get(field, "")) != str(expected_metadata.get(field, "")):
            return False
    return True


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
        self.enable_think_pass = gemma_config.get("enable_think_pass", True)
        self.force_think_all = bool(gemma_config.get("force_think_all", False))
        self.use_inputs_embeds_for_forward = gemma_config.get(
            "use_inputs_embeds_for_forward", False
        )
        self.use_language_model_direct = gemma_config.get("use_language_model_direct", False)
        self.conf_low = float(gemma_config.get("think_conf_low", 0.35))
        self.conf_high = float(gemma_config.get("think_conf_high", 0.65))
        self.max_think_tokens = gemma_config["max_think_tokens"]
        self.chat_template_enable_thinking_fast = gemma_config.get(
            "chat_template_enable_thinking_fast"
        )
        self.chat_template_enable_thinking_think = gemma_config.get(
            "chat_template_enable_thinking_think"
        )
        self.debug_log_path = "logs/debug_llm_verifier.jsonl"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_device = None
        self.model = None
        self.processor = None
        self.tokenizer = None

        self.token_f_ids = []
        self.token_h_ids = []

        self.exemplar_retriever = ExemplarRetriever(self.config)

    def cache_metadata(self):
        return {
            "model_name": self.model_name,
            "model_loader": self.model_loader,
            "load_in": self.load_in,
            "max_input_tokens": self.max_input_tokens,
            "enable_think_pass": self.enable_think_pass,
            "force_think_all": self.force_think_all,
            "exemplar_top_k": self.exemplar_top_k,
            "think_conf_low": self.conf_low,
            "think_conf_high": self.conf_high,
            "chat_template_enable_thinking_fast": self.chat_template_enable_thinking_fast,
            "chat_template_enable_thinking_think": self.chat_template_enable_thinking_think,
        }

    def load_model(self):
        from src.config_utils import resolve_model_path

        resolved_name = resolve_model_path(self.model_name)
        info(f"Loading verifier model: {self.model_name}")
        info(f"Device={self.device} · load_in={self.load_in} · loader={self.model_loader}")
        if self.clear_cuda_before_load and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        with console.status("Initializing processor...", spinner="aesthetic"):
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

        with console.status(
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

        ok(f"Verifier loaded · input_device={self.input_device}")

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

    def _think_instruction(self, task_type: str):
        """English reasoning instruction for the think pass."""
        return (
            "Instruction: Assess if candidate response 'A' to question 'Q' is correct and faithful (verdict: Faithful) "
            "or wrong/hallucinated (verdict: Hallucinated) based on the Rule and Evidence.\n"
            "At the end of your response, write exactly this format:\n"
            "verdict: Faithful|Hallucinated\n"
            "confidence: strong|likely|uncertain\n"
            "reason: <one short English sentence>"
        )

    def _think_token_budget(self, task_type: str, think_reasons: list[str]):
        """Use the configured cap to ensure the model has enough room to reason and format its verdict."""
        return int(self.max_think_tokens)

    def _apply_chat_template(self, messages, *, enable_thinking=None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            return self.processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.processor.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _parse_think_output(generated_text: str):
        verdict = None
        verdict_match = re.search(
            r"verdict\s*[:=is\s]*\s*(faithful|hallucinated|correct|wrong|truthful|false|f|h)",
            generated_text,
            re.IGNORECASE,
        )
        if verdict_match:
            val = verdict_match.group(1).lower()
            if val in ("faithful", "correct", "truthful", "f"):
                verdict = "Faithful"
            elif val in ("hallucinated", "wrong", "false", "h"):
                verdict = "Hallucinated"

        if not verdict:
            last_part = generated_text[-150:].lower()
            if "faithful" in last_part or "correct" in last_part:
                verdict = "Faithful"
            elif "hallucinated" in last_part or "wrong" in last_part:
                verdict = "Hallucinated"

        confidence_match = re.search(
            r"confidence\s*[:=is\s]*\s*(strong|likely|uncertain|high|medium|low)",
            generated_text,
            re.IGNORECASE,
        )
        confidence = "likely"
        if confidence_match:
            val = confidence_match.group(1).lower()
            if val in ("strong", "high"):
                confidence = "strong"
            elif val in ("uncertain", "low"):
                confidence = "uncertain"
            else:
                confidence = "likely"

        score = map_think_verdict(verdict, confidence) if verdict else None
        return verdict, confidence, score

    def predict_single(
        self,
        evidence,
        prompt_bn,
        response_bn,
        task_type=None,
        context_original=None,
        has_context=None,
        silent=True,
        write_log=True,
    ):
        if self.model is None:
            self.load_model()

        if task_type is None:
            ctx_for_route = (
                context_original
                if context_original is not None
                else (evidence if has_context else "[NULL]")
            )
            task_type = route_row(ctx_for_route or "[NULL]", prompt_bn, response_bn)
        if context_original is None:
            context_original = evidence if has_context else "[NULL]"

        instruction = task_instruction(task_type)

        if self.exemplar_top_k > 0:
            exemplars = self.exemplar_retriever.retrieve_exemplars(
                query=f"{prompt_bn} {response_bn}",
                exclude_prompt=prompt_bn,
                exclude_response=response_bn,
                top_k=self.exemplar_top_k,
            )
        else:
            exemplars = []

        prompt_exemplars = ""
        for idx, ex in enumerate(exemplars):
            ex_label_str = "F" if ex["label"] == 1 else "H"
            prompt_exemplars += (
                f"Ex {idx + 1}\n"
                f"E: {ex['context']}\n"
                f"Q: {ex['prompt_bn']}\n"
                f"A: {ex['response_bn']}\n"
                f"V: {ex_label_str}\n\n"
            )

        evidence_clean = str(evidence).strip()
        if evidence_clean in ("", "[NULL]", "None", "nan"):
            evidence_str = "No evidence provided. Use your internal knowledge to judge the correctness of the answer based on the Rule."
        else:
            evidence_str = f"<evidence>\n{evidence_clean}\n</evidence>"

        user_content = (
            f"{prompt_exemplars}"
            f"Task: {task_type}\n"
            f"Rule: {instruction}\n"
            f"{evidence_str}\n"
            f"Q: {prompt_bn}\n"
            f"A: {response_bn}\n"
            f"Return one token only: F = faithful/correct/label 1; "
            f"H = hallucinated/wrong/label 0.\n"
        )

        messages = [{"role": "user", "content": user_content}]
        prompt = self._apply_chat_template(
            messages,
            enable_thinking=self.chat_template_enable_thinking_fast,
        )
        prompt += "V:"

        inputs = self._prepare_inputs(prompt)

        with torch.inference_mode():
            logits = self._forward_logits(inputs)

        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

        prob_f = sum(probs[tid].item() for tid in self.token_f_ids if tid is not None)
        prob_h = sum(probs[tid].item() for tid in self.token_h_ids if tid is not None)

        sum_prob = prob_f + prob_h
        p_f = (prob_f / sum_prob) if sum_prob > 0 else 0.5
        p_llm = p_f
        p_fast = p_f

        think_reasons = []
        triggered_think = should_trigger_think(
            p_fast=p_llm,
            task_type=task_type,
            evidence=evidence,
            context_original=context_original,
            prompt_bn=prompt_bn,
            conf_low=self.conf_low,
            conf_high=self.conf_high,
            think_reasons=think_reasons,
            force_think_all=self.force_think_all,
        )
        p_llm_no_think = p_llm
        generated_text = ""
        verdict_parsed = None
        confidence_parsed = None
        think_max_tokens = None

        if triggered_think and not self.enable_think_pass:
            think_reasons.append("think_pass_disabled")
            triggered_think = False

        if triggered_think:
            if not silent:
                reason_str = ", ".join(think_reasons) if think_reasons else "unknown"
                warn(f"Think pass triggered ({reason_str}, p_fast={p_llm:.4f})")

            think_user_content = (
                f"{prompt_exemplars}"
                f"Task: {task_type}\n"
                f"Rule: {instruction}\n"
                f"{evidence_str}\n"
                f"Q: {prompt_bn}\n"
                f"A: {response_bn}\n"
                f"{self._think_instruction(task_type)}"
            )

            think_messages = [{"role": "user", "content": think_user_content}]
            think_prompt = self._apply_chat_template(
                think_messages,
                enable_thinking=self.chat_template_enable_thinking_think,
            )

            think_inputs = self._prepare_inputs(think_prompt, use_inputs_embeds=False)
            think_max_tokens = self._think_token_budget(task_type, think_reasons)

            with torch.inference_mode():
                gen_outputs = self.model.generate(
                    **think_inputs,
                    max_new_tokens=think_max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            input_len = think_inputs["input_ids"].shape[1]
            generated_tokens = gen_outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            verdict, confidence, think_score = self._parse_think_output(generated_text)
            verdict_parsed = verdict
            confidence_parsed = confidence
            if think_score is not None:
                p_llm = think_score
            else:
                think_reasons.append("verdict_unparsed")
                if not silent:
                    warn(
                        f"Think pass produced no verdict in {think_max_tokens} "
                        f"tokens; keeping p_fast={p_llm:.4f}"
                    )

        generated_text_log = generated_text if triggered_think else None

        log_entry = {
            "cache_version": CACHE_VERSION,
            "evidence": evidence,
            "context_original": context_original,
            "prompt": prompt_bn,
            "response": response_bn,
            "task_type": task_type,
            "has_context": None if has_context is None else bool(has_context),
            "p_fast": float(p_fast),
            "p_think": None if not triggered_think else float(p_llm),
            "p_llm_no_think": float(p_llm_no_think),
            "triggered_think": bool(triggered_think),
            "think_reasons": think_reasons,
            "think_max_tokens": think_max_tokens,
            "thinking_cot": generated_text_log,
            "verdict_parsed": verdict_parsed,
            "confidence_parsed": confidence_parsed,
            "p_llm_final": float(p_llm),
        }
        log_entry.update(self.cache_metadata())
        if write_log:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            with open(self.debug_log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return p_llm, triggered_think

    @staticmethod
    def _emit_partial(on_partial, n_done, preds):
        try:
            on_partial(n_done, list(preds))
        except Exception as e:
            warn(f"Partial submission write failed ({e}); continuing")

    def predict_dataset(
        self,
        df,
        use_cache=True,
        debug_log_path=None,
        on_partial=None,
        partial_every=0,
    ):
        """Score every row with the verifier.

        on_partial(n_done, preds_so_far) is invoked every `partial_every` rows (and
        once at the end) so callers can persist partial output; a failure inside it
        must not kill an in-flight run, so it is called defensively.
        """
        active_log_path = debug_log_path or self.debug_log_path
        original_log_path = self.debug_log_path
        self.debug_log_path = active_log_path

        cache = {}
        cache_metadata = self.cache_metadata()
        if use_cache and os.path.exists(active_log_path):
            try:
                with open(active_log_path, "r", encoding="utf-8") as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if not verifier_log_matches_metadata(entry, cache_metadata):
                                continue
                            key = verifier_case_key(
                                evidence=entry.get("evidence", ""),
                                prompt=entry.get("prompt", ""),
                                response=entry.get("response", ""),
                                task_type=entry.get("task_type", ""),
                                context_original=entry.get("context_original", ""),
                                metadata=cache_metadata,
                            )
                            cache[key] = (
                                entry["p_llm_final"],
                                entry["triggered_think"],
                            )
                        except Exception:
                            continue
                if cache:
                    ok(f"Loaded {len(cache)} cached predictions from debug log ({active_log_path})")
            except Exception as e:
                warn(f"Could not read existing debug log: {e}. Starting fresh.")

        if "task_type" in df.columns:
            task_types = df["task_type"].astype(str).tolist()
        else:
            task_types = [
                route_row(
                    row.get("context_original", row["context"]),
                    row["prompt_bn"],
                    row.get("response_bn", ""),
                )
                for _, row in df.iterrows()
            ]

        if "context_original" in df.columns:
            context_originals = df["context_original"].astype(str).tolist()
        else:
            context_originals = df["context"].astype(str).tolist()

        if "has_context" in df.columns:
            has_context_values = df["has_context"].values
        else:
            has_context_values = [
                str(c).strip() not in ("[NULL]", "", "None", "nan") for c in context_originals
            ]

        preds = []
        total_rows = len(df)
        think_count = 0
        cache_hits = 0

        try:
            with pipeline_progress() as progress:
                task = progress.add_task("Verifier", total=total_rows)

                for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
                    evidence = str(row["context"])
                    prompt = str(row["prompt_bn"])
                    response = str(row["response_bn"])
                    row_task_type = task_types[row_num - 1]
                    row_context_original = context_originals[row_num - 1]
                    row_has_context = bool(has_context_values[row_num - 1])
                    key = verifier_case_key(
                        evidence=evidence,
                        prompt=prompt,
                        response=response,
                        task_type=row_task_type,
                        context_original=row_context_original,
                        metadata=cache_metadata,
                    )

                    if use_cache and key in cache:
                        prob, triggered_think = cache[key]
                        cache_hits += 1
                    else:
                        if self.model is None:
                            self.load_model()
                        prob, triggered_think = self.predict_single(
                            evidence,
                            prompt,
                            response,
                            task_type=row_task_type,
                            context_original=row_context_original,
                            has_context=row_has_context,
                            silent=True,
                            write_log=use_cache,
                        )

                    preds.append(prob)

                    if triggered_think:
                        think_count += 1

                    progress.update(
                        task,
                        description=(
                            f"Verifier · {row_task_type} · think {think_count} · cache {cache_hits}"
                        ),
                    )
                    progress.advance(task)

                    if on_partial and partial_every and row_num % partial_every == 0:
                        self._emit_partial(on_partial, row_num, preds)
                        info(f"Partial debug flushed at row {row_num}/{total_rows}")
        finally:
            self.debug_log_path = original_log_path

        if on_partial and partial_every:
            self._emit_partial(on_partial, len(preds), preds)

        ok(
            f"Verifier done · think {think_count}/{total_rows} "
            f"({100.0 * think_count / max(total_rows, 1):.1f}%) · "
            f"cache hits {cache_hits}"
        )
        return np.array(preds)


def main():
    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)

    train_path = os.path.join(config["data"]["processed_dir"], "train.csv")
    if not os.path.exists(train_path):
        warn("Preprocessed train data not found. Run `just preprocess` first.")
        return

    df = pd.read_csv(train_path)
    df_sample = df.head(3)

    banner("Gemma verifier smoke", "3-row sample from processed train.csv")
    verifier = GemmaVerifier()
    preds = verifier.predict_dataset(df_sample)
    info(f"Sample predictions: {preds}")


if __name__ == "__main__":
    main()
