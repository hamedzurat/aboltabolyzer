import json
import logging
import os
import pickle
import re
import time
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
from src.evidence_policy import should_trigger_think, task_instruction
from src.router import route_row
from src.tui import banner, console, info, ok, pipeline_progress, warn

# Suppress Hugging Face warnings/load reports for a cleaner UI
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
disable_progress_bars()
logging.getLogger("transformers").setLevel(logging.ERROR)

CACHE_VERSION = "verifier-cache-v3"
CACHE_METADATA_FIELDS = (
    "fast_model_name",
    "think_model_name",
    "model_loader",
    "load_in",
    "max_input_tokens",
    "enable_think_pass",
    "exemplar_top_k",
    "think_conf_low",
    "think_conf_high",
    "chat_template_enable_thinking_fast",
    "chat_template_enable_thinking_think",
    "fast_pass_batch_size",
    "think_pass_batch_size",
    "max_think_tokens",
    "max_think_tokens_by_task",
    "nli_cache_tag",
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

    def retrieve_exemplars(
        self,
        query,
        exclude_prompt=None,
        exclude_response=None,
        top_k=3,
        query_emb=None,
    ):
        """Retrieves top_k nearest exemplars, avoiding target leakage by filtering out exact matching inputs."""
        if self.embeddings is None:
            if not self.load_index():
                return []

        if query_emb is None:
            self.load_model()
            query_emb = self.model.encode(
                [query], show_progress_bar=False, normalize_embeddings=True
            )[0]

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

        self.fast_model_name = gemma_config.get("fast_model_name", gemma_config.get("model_name"))
        self.think_model_name = gemma_config.get("think_model_name", gemma_config.get("model_name"))
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
        self.max_think_tokens = int(gemma_config["max_think_tokens"])
        raw_by_task = gemma_config.get("max_think_tokens_by_task") or {}
        self.max_think_tokens_by_task = {str(k): int(v) for k, v in dict(raw_by_task).items()}
        self.chat_template_enable_thinking_fast = gemma_config.get(
            "chat_template_enable_thinking_fast"
        )
        self.chat_template_enable_thinking_think = gemma_config.get(
            "chat_template_enable_thinking_think"
        )
        self.fast_pass_batch_size = int(gemma_config.get("fast_pass_batch_size", 8))
        self.think_pass_batch_size = int(gemma_config.get("think_pass_batch_size", 1))
        self.debug_log_path = "logs/debug_llm_verifier.jsonl"
        self._nli_cache_tag = "nli_off"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_device = None
        self.model = None
        self.processor = None
        self.tokenizer = None

        self.token_f_ids = []
        self.token_h_ids = []

        self.exemplar_retriever = ExemplarRetriever(self.config)

    def cache_metadata(self):
        by_task = ",".join(f"{k}:{v}" for k, v in sorted(self.max_think_tokens_by_task.items()))
        return {
            "fast_model_name": self.fast_model_name,
            "think_model_name": self.think_model_name,
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
            "fast_pass_batch_size": self.fast_pass_batch_size,
            "think_pass_batch_size": self.think_pass_batch_size,
            "max_think_tokens": self.max_think_tokens,
            "max_think_tokens_by_task": by_task,
            "nli_cache_tag": self._nli_cache_tag,
        }

    def load_model(self, model_name=None):
        from src.config_utils import resolve_model_path

        if model_name is None:
            model_name = self.fast_model_name

        if self.model is not None and getattr(self, "_loaded_model_name", None) == model_name:
            return

        resolved_name = resolve_model_path(model_name)
        info(f"Loading verifier model: {model_name}")
        self._loaded_model_name = model_name
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
            try:
                self.model = model_class.from_pretrained(
                    resolved_name, attn_implementation="flash_attention_2", **load_kwargs
                )
            except Exception:
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
        partial_ok_note = (
            "A correct partial answer is Faithful — do NOT mark Hallucinated "
            "just because the answer omits details the evidence did not ask for."
            if task_type
            in ("context_grounded_fact", "context_grounded_other", "famous_bn_fact_context")
            else ""
        )
        math_note = (
            "Show your calculation step by step, then state the verdict."
            if task_type
            in (
                "math_work_rate",
                "math_speed_distance",
                "math_profit_loss",
                "math_average",
                "calendar_arithmetic",
                "translation_or_bilingual",
            )
            else ""
        )
        extra = " ".join(p for p in [partial_ok_note, math_note] if p)
        return (
            "Instruction: Assess if candidate response 'A' to question 'Q' is correct and "
            "faithful (verdict: Faithful) or wrong/hallucinated (verdict: Hallucinated) "
            "based on the Rule and Evidence.\n"
            + (f"{extra}\n" if extra else "")
            + "Reason briefly, then end with exactly:\n"
            "verdict: Faithful|Hallucinated"
        )

    def _think_token_budget(self, task_type: str, think_reasons: list[str] | None = None):
        """Per-task think budget from config; falls back to max_think_tokens."""
        del think_reasons  # kept for call-site compatibility / future use
        if task_type in self.max_think_tokens_by_task:
            return int(self.max_think_tokens_by_task[task_type])
        return int(self.max_think_tokens)

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        pos = attention_mask.long().cumsum(dim=-1) - 1
        return pos.masked_fill(attention_mask == 0, 0)

    def _evidence_str(self, evidence) -> str:
        evidence_clean = str(evidence).strip()
        if evidence_clean in ("", "[NULL]", "None", "nan"):
            return (
                "No evidence provided. Use your internal knowledge to judge the "
                "correctness of the answer based on the Rule."
            )
        return f"<evidence>\n{evidence_clean}\n</evidence>"

    def _exemplar_block(self, exemplars) -> str:
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
        return prompt_exemplars

    def _retrieve_exemplars(self, prompt_bn, response_bn, query_emb=None):
        if self.exemplar_top_k <= 0:
            return []
        return self.exemplar_retriever.retrieve_exemplars(
            query=f"{prompt_bn} {response_bn}",
            exclude_prompt=prompt_bn,
            exclude_response=response_bn,
            top_k=self.exemplar_top_k,
            query_emb=query_emb,
        )

    def _build_think_prompt(self, *, evidence, prompt_bn, response_bn, task_type, exemplars):
        instruction = task_instruction(task_type)
        think_user_content = (
            f"{self._exemplar_block(exemplars)}"
            f"Task: {task_type}\n"
            f"Rule: {instruction}\n"
            f"{self._evidence_str(evidence)}\n"
            f"Q: {prompt_bn}\n"
            f"A: {response_bn}\n"
            f"{self._think_instruction(task_type)}"
        )
        think_messages = [{"role": "user", "content": think_user_content}]
        return self._apply_chat_template(
            think_messages,
            enable_thinking=self.chat_template_enable_thinking_think,
        )

    def _batched_think_generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Left-padded batched greedy generate for the think pass."""
        if not prompts:
            return []
        if len(prompts) == 1:
            think_inputs = self._prepare_inputs(prompts[0], use_inputs_embeds=False)
            with torch.inference_mode():
                gen_outputs = self.model.generate(
                    **think_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            input_len = think_inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(gen_outputs[0][input_len:], skip_special_tokens=True)
            return [text]

        tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        original_padding_side = tokenizer.padding_side
        original_truncation_side = getattr(tokenizer, "truncation_side", None)
        tokenizer.padding_side = "left"
        if self.max_input_tokens is not None:
            tokenizer.truncation_side = "left"
        try:
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=self.max_input_tokens is not None,
                max_length=self.max_input_tokens,
            )
        finally:
            tokenizer.padding_side = original_padding_side
            if original_truncation_side is not None:
                tokenizer.truncation_side = original_truncation_side

        target_device = self.input_device or self.model.device
        enc = {k: v.to(target_device) for k, v in enc.items()}
        enc["position_ids"] = self._position_ids_from_mask(enc["attention_mask"])
        pad_width = int(enc["input_ids"].shape[1])

        with torch.inference_mode():
            gen_outputs = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        texts = []
        for i in range(len(prompts)):
            texts.append(
                self.tokenizer.decode(gen_outputs[i][pad_width:], skip_special_tokens=True)
            )
        return texts

    def _append_debug_log(self, log_entry: dict):
        os.makedirs(os.path.dirname(self.debug_log_path) or ".", exist_ok=True)
        with open(self.debug_log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

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

        confidence = None
        score = None
        if verdict == "Faithful":
            score = 0.90
        elif verdict == "Hallucinated":
            score = 0.10

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
        query_emb=None,
        p_fast=None,
        force_think=None,
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

        exemplars = self._retrieve_exemplars(prompt_bn, response_bn, query_emb=query_emb)
        prompt_exemplars = self._exemplar_block(exemplars)
        evidence_str = self._evidence_str(evidence)

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

        if p_fast is not None:
            p_llm = p_fast
        else:
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
        if force_think is not None:
            triggered_think = force_think
        else:
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

            think_prompt = self._build_think_prompt(
                evidence=evidence,
                prompt_bn=prompt_bn,
                response_bn=response_bn,
                task_type=task_type,
                exemplars=exemplars,
            )
            think_max_tokens = self._think_token_budget(task_type, think_reasons)
            generated_text = self._batched_think_generate([think_prompt], think_max_tokens)[0]

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
            self._append_debug_log(log_entry)

        return p_llm, triggered_think

    @staticmethod
    def _emit_partial(on_partial, n_done, preds):
        try:
            on_partial(n_done, list(preds))
        except Exception as e:
            warn(f"Partial submission write failed ({e}); continuing")

    def route_single_llm(self, prompt_bn: str, response_bn: str = "") -> str:
        """Classify a row's task category using a single-token forward pass (LLM routing)."""
        if self.model is None:
            self.load_model()

        user_content = (
            "Classify this Bengali question 'Q' and candidate answer 'A' into one of these categories:\n"
            "G: Grammar/Syntax/Spelling rules (কারক, বিভক্তি, সন্ধি, সমাস, বানান, ইত্যাদি)\n"
            "M: Mathematics, numbers, averages, speed, profit, or calendar arithmetic (গড়, লাভ-ক্ষতি, গতিবেগ, কাজ-সময়, ক্যালেন্ডার, ইত্যাদি)\n"
            "I: Idioms (বাগধারা/প্রবাদ-প্রবচন) or figurative/compositional meaning (শাব্দিক অর্থ, ইত্যাদি)\n"
            "T: Translation, English-Bengali bilingual terms, or definition comparison\n"
            "F: Factual knowledge (General knowledge, geography, famous Bengali history/literature/entities)\n"
            "O: Other general questions/uncategorized\n\n"
            f"Q: {prompt_bn}\n"
            f"A: {response_bn}\n\n"
            "Return exactly one character category code (G, M, I, T, F, or O):\n"
            "Category:"
        )

        messages = [{"role": "user", "content": user_content}]
        prompt = self._apply_chat_template(
            messages,
            enable_thinking=False,
        )

        routing_chars = ["G", "M", "I", "T", "F", "O"]
        tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
        char_to_id = {}
        for char in routing_chars:
            tids = tokenizer.encode(char, add_special_tokens=False)
            if tids:
                char_to_id[char] = tids[-1]

        inputs = self._prepare_inputs(prompt)
        with torch.inference_mode():
            logits = self._forward_logits(inputs)

        next_token_logits = logits[0, -1, :]
        char_logits = {
            char: next_token_logits[tid].item()
            for char, tid in char_to_id.items()
            if tid is not None
        }

        best_char = max(char_logits, key=char_logits.get) if char_logits else "O"
        return best_char

    def predict_dataset(
        self,
        df,
        use_cache=True,
        debug_log_path=None,
        on_partial=None,
        partial_every=0,
        partial_every_seconds=0,
        nli_config=None,
    ):
        """Score rows: batched fast → NLI-first gate → think fallback."""
        from src.nli import NLIRefiner, empty_nli_debug, nli_cache_tag, release_cuda_for_nli

        # Prefer seconds; legacy partial_every (row count) → once a minute
        if partial_every_seconds and float(partial_every_seconds) > 0:
            flush_every_s = float(partial_every_seconds)
        elif partial_every and int(partial_every) > 0:
            flush_every_s = 60.0
        else:
            flush_every_s = 0.0
        partial_state = {"last_at": time.monotonic()}

        self._nli_cache_tag = nli_cache_tag(nli_config)
        active_log_path = debug_log_path or self.debug_log_path
        original_log_path = self.debug_log_path
        self.debug_log_path = active_log_path
        # Ensure logs/ exists immediately so mid-run checkpoints are visible
        os.makedirs(os.path.dirname(active_log_path) or "logs", exist_ok=True)

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

        total_rows = len(df)
        query_embs = [None] * total_rows
        if self.exemplar_top_k > 0:
            self.exemplar_retriever.load_model()
            queries = [f"{row['prompt_bn']} {row['response_bn']}" for _, row in df.iterrows()]
            embs = self.exemplar_retriever.model.encode(
                queries,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32,
            )
            query_embs = list(embs)

        # 1. Batched Fast Pass
        p_fast_vals = [None] * total_rows
        triggered_think_vals = [False] * total_rows
        fast_pass_needed = []

        for i, (_idx, r) in enumerate(df.iterrows()):
            evidence = str(r["context"])
            prompt = str(r["prompt_bn"])
            response = str(r["response_bn"])
            key = verifier_case_key(
                evidence=evidence,
                prompt=prompt,
                response=response,
                task_type=task_types[i],
                context_original=context_originals[i],
                metadata=cache_metadata,
            )
            if use_cache and key in cache:
                prob, trig = cache[key]
                p_fast_vals[i] = prob
                triggered_think_vals[i] = trig
            else:
                fast_pass_needed.append((i, r))

        if fast_pass_needed:
            self.load_model(self.fast_model_name)
            tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            batch_size = self.fast_pass_batch_size
            with pipeline_progress() as progress:
                fast_task = progress.add_task("Verifier (Fast Pass)", total=len(fast_pass_needed))
                for start_idx in range(0, len(fast_pass_needed), batch_size):
                    batch = fast_pass_needed[start_idx : start_idx + batch_size]
                    prompts = []
                    for orig_i, r in batch:
                        exemplars = []
                        if self.exemplar_top_k > 0:
                            exemplars = self.exemplar_retriever.retrieve_exemplars(
                                query=f"{r['prompt_bn']} {r['response_bn']}",
                                exclude_prompt=r["prompt_bn"],
                                exclude_response=r["response_bn"],
                                top_k=self.exemplar_top_k,
                                query_emb=query_embs[orig_i],
                            )
                        prompt_exemplars = self._exemplar_block(exemplars)
                        evidence_str = self._evidence_str(r["context"])
                        instruction = task_instruction(task_types[orig_i])
                        user_content = (
                            f"{prompt_exemplars}Task: {task_types[orig_i]}\n"
                            f"Rule: {instruction}\n{evidence_str}\n"
                            f"Q: {r['prompt_bn']}\nA: {r['response_bn']}\n"
                            "Return one token only: F = faithful/correct/label 1; "
                            "H = hallucinated/wrong/label 0.\n"
                        )
                        messages = [{"role": "user", "content": user_content}]
                        prompts.append(
                            self._apply_chat_template(
                                messages,
                                enable_thinking=self.chat_template_enable_thinking_fast,
                            )
                            + "V:"
                        )

                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_tokens,
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    with torch.inference_mode():
                        logits = self._forward_logits(inputs)

                    for idx_in_batch, (orig_i, r) in enumerate(batch):
                        seq_len = int(inputs["attention_mask"][idx_in_batch].sum().item())
                        seq_len_dim = logits.shape[1]
                        target_seq_idx = 0 if seq_len_dim == 1 else (seq_len - 1)
                        next_token_logits = logits[idx_in_batch, target_seq_idx, :]
                        probs = torch.softmax(next_token_logits, dim=-1)
                        prob_f = sum(
                            probs[tid].item() for tid in self.token_f_ids if tid is not None
                        )
                        prob_h = sum(
                            probs[tid].item() for tid in self.token_h_ids if tid is not None
                        )
                        sum_prob = prob_f + prob_h
                        p_f = (prob_f / sum_prob) if sum_prob > 0 else 0.5
                        p_fast_vals[orig_i] = p_f
                        think_reasons = []
                        triggered_think_vals[orig_i] = should_trigger_think(
                            p_fast=p_f,
                            task_type=task_types[orig_i],
                            evidence=str(r["context"]),
                            context_original=context_originals[orig_i],
                            prompt_bn=str(r["prompt_bn"]),
                            conf_low=self.conf_low,
                            conf_high=self.conf_high,
                            think_reasons=think_reasons,
                            force_think_all=self.force_think_all,
                        )
                    progress.advance(fast_task, len(batch))

        # 2. NLI-first gate
        nli_debug = empty_nli_debug(df.index)
        nli_enabled = bool(nli_config and nli_config.get("enabled"))
        nli_skipped_think = 0
        if nli_enabled and fast_pass_needed:
            info("Unloading verifier before NLI-first gate...")
            self.model = None
            self.tokenizer = None
            self.processor = None
            release_cuda_for_nli()

            uncached_indices = []
            for i in range(total_rows):
                r = df.iloc[i]
                key = verifier_case_key(
                    evidence=str(r["context"]),
                    prompt=str(r["prompt_bn"]),
                    response=str(r["response_bn"]),
                    task_type=task_types[i],
                    context_original=context_originals[i],
                    metadata=cache_metadata,
                )
                if not (use_cache and key in cache):
                    uncached_indices.append(i)

            refiner = NLIRefiner(nli_config)
            with pipeline_progress() as progress:
                nli_task = progress.add_task("NLI-first gate", total=max(len(uncached_indices), 1))
                if uncached_indices:
                    gate_df = df.iloc[uncached_indices].copy()
                    scored = {"n": 0}

                    def _adv(n):
                        scored["n"] += n
                        progress.advance(nli_task, n)

                    partial = refiner.gate(gate_df, on_batch=_adv)
                    for gidx in partial.index:
                        nli_debug.loc[gidx] = partial.loc[gidx]
                    # Fill remainder so the bar completes (eligible ⊂ uncached)
                    left = max(len(uncached_indices), 1) - scored["n"]
                    if left > 0:
                        progress.advance(nli_task, left)
                else:
                    progress.advance(nli_task, 1)
            refiner.unload()
            release_cuda_for_nli()

            for i in range(total_rows):
                idx = df.index[i]
                if bool(nli_debug.at[idx, "nli_applied"]):
                    if triggered_think_vals[i]:
                        nli_skipped_think += 1
                    triggered_think_vals[i] = False
        elif nli_enabled and not fast_pass_needed:
            info("NLI-first: all rows cached — skipping gate")

        # 3. Think jobs (after NLI may clear triggers)
        think_jobs = []
        for i in range(total_rows):
            r = df.iloc[i]
            key = verifier_case_key(
                evidence=str(r["context"]),
                prompt=str(r["prompt_bn"]),
                response=str(r["response_bn"]),
                task_type=task_types[i],
                context_original=context_originals[i],
                metadata=cache_metadata,
            )
            if use_cache and key in cache:
                continue
            if not triggered_think_vals[i] or not self.enable_think_pass:
                continue
            think_reasons = []
            should_trigger_think(
                p_fast=p_fast_vals[i],
                task_type=task_types[i],
                evidence=str(r["context"]),
                context_original=context_originals[i],
                prompt_bn=str(r["prompt_bn"]),
                conf_low=self.conf_low,
                conf_high=self.conf_high,
                think_reasons=think_reasons,
                force_think_all=self.force_think_all,
            )
            think_jobs.append((i, r, think_reasons))

        think_pass_todo = len(think_jobs)
        if think_pass_todo > 0:
            self.load_model(self.think_model_name)

        preds = [None] * total_rows
        think_count = 0
        cache_hits = 0
        nli_applied_count = 0

        def _maybe_partial(*, force=False):
            if not on_partial or flush_every_s <= 0:
                return
            now = time.monotonic()
            if not force and (now - partial_state["last_at"]) < flush_every_s:
                return
            n_ready = sum(1 for p in preds if p is not None)
            if n_ready == 0:
                return
            # Pass full-length preds (None = not ready) so the writer can align by row
            self._emit_partial(on_partial, n_ready, preds)
            partial_state["last_at"] = now

        def _nli_fields(idx):
            row = nli_debug.loc[idx]
            return {
                "nli_eligible": bool(row["nli_eligible"]),
                "nli_applied": bool(row["nli_applied"]),
                "nli_skip_reason": row["nli_skip_reason"] or "",
                "nli_p_entail": (
                    None if pd.isna(row["nli_p_entail"]) else float(row["nli_p_entail"])
                ),
                "nli_p_contradict": (
                    None
                    if pd.isna(row["nli_p_contradict"])
                    else float(row["nli_p_contradict"])
                ),
                "nli_p_neutral": (
                    None if pd.isna(row["nli_p_neutral"]) else float(row["nli_p_neutral"])
                ),
                "nli_margin": (None if pd.isna(row["nli_margin"]) else float(row["nli_margin"])),
                "p_nli": None if pd.isna(row["p_nli"]) else float(row["p_nli"]),
            }

        def _finalize_row(i, *, generated_text=None, think_max_tokens=None, think_reasons=None):
            """Fill preds[i], append jsonl, for one uncached row."""
            nonlocal think_count, cache_hits, nli_applied_count
            r = df.iloc[i]
            evidence = str(r["context"])
            prompt = str(r["prompt_bn"])
            response = str(r["response_bn"])
            row_task_type = task_types[i]
            row_context_original = context_originals[i]
            row_has_context = bool(has_context_values[i])
            idx = df.index[i]
            key = verifier_case_key(
                evidence=evidence,
                prompt=prompt,
                response=response,
                task_type=row_task_type,
                context_original=row_context_original,
                metadata=cache_metadata,
            )
            if use_cache and key in cache:
                preds[i] = cache[key][0]
                cache_hits += 1
                return

            p_fast = p_fast_vals[i]
            nli_meta = _nli_fields(idx)

            if p_fast is None:
                prob, triggered_think = self.predict_single(
                    evidence,
                    prompt,
                    response,
                    task_type=row_task_type,
                    context_original=row_context_original,
                    has_context=row_has_context,
                    silent=True,
                    write_log=use_cache,
                    query_emb=query_embs[i],
                )
                preds[i] = prob
                if triggered_think:
                    think_count += 1
                return

            if nli_meta["nli_applied"] and nli_meta["p_nli"] is not None:
                p_llm = float(nli_meta["p_nli"])
                preds[i] = p_llm
                nli_applied_count += 1
                if use_cache:
                    log_entry = {
                        "cache_version": CACHE_VERSION,
                        "evidence": evidence,
                        "context_original": row_context_original,
                        "prompt": prompt,
                        "response": response,
                        "task_type": row_task_type,
                        "has_context": row_has_context,
                        "p_fast": float(p_fast),
                        "p_think": None,
                        "p_llm_no_think": float(p_fast),
                        "triggered_think": False,
                        "think_reasons": ["nli_confident_skip_think"],
                        "think_max_tokens": None,
                        "thinking_cot": None,
                        "verdict_parsed": None,
                        "confidence_parsed": None,
                        "p_llm_final": float(p_llm),
                        **nli_meta,
                    }
                    log_entry.update(self.cache_metadata())
                    self._append_debug_log(log_entry)
                return

            if generated_text is not None:
                p_llm = float(p_fast)
                p_llm_no_think = p_llm
                reasons = list(think_reasons or [])
                verdict, confidence, think_score = self._parse_think_output(generated_text)
                if think_score is not None:
                    p_llm = think_score
                else:
                    reasons.append("verdict_unparsed")
                preds[i] = p_llm
                think_count += 1
                if use_cache:
                    log_entry = {
                        "cache_version": CACHE_VERSION,
                        "evidence": evidence,
                        "context_original": row_context_original,
                        "prompt": prompt,
                        "response": response,
                        "task_type": row_task_type,
                        "has_context": row_has_context,
                        "p_fast": float(p_fast),
                        "p_think": float(p_llm),
                        "p_llm_no_think": float(p_llm_no_think),
                        "triggered_think": True,
                        "think_reasons": reasons,
                        "think_max_tokens": think_max_tokens,
                        "thinking_cot": generated_text,
                        "verdict_parsed": verdict,
                        "confidence_parsed": confidence,
                        "p_llm_final": float(p_llm),
                        **nli_meta,
                    }
                    log_entry.update(self.cache_metadata())
                    self._append_debug_log(log_entry)
                return

            # Fast-only
            preds[i] = float(p_fast)
            if use_cache:
                reasons = []
                if triggered_think_vals[i] and not self.enable_think_pass:
                    reasons.append("think_pass_disabled")
                log_entry = {
                    "cache_version": CACHE_VERSION,
                    "evidence": evidence,
                    "context_original": row_context_original,
                    "prompt": prompt,
                    "response": response,
                    "task_type": row_task_type,
                    "has_context": row_has_context,
                    "p_fast": float(p_fast),
                    "p_think": None,
                    "p_llm_no_think": float(p_fast),
                    "triggered_think": False,
                    "think_reasons": reasons,
                    "think_max_tokens": None,
                    "thinking_cot": None,
                    "verdict_parsed": None,
                    "confidence_parsed": None,
                    "p_llm_final": float(p_fast),
                    **nli_meta,
                }
                log_entry.update(self.cache_metadata())
                self._append_debug_log(log_entry)

        # Finalize non-think rows now so logs/ + partial appear before long think
        think_index_set = {job[0] for job in think_jobs}
        for i in range(total_rows):
            if i in think_index_set:
                continue
            if preds[i] is not None:
                continue
            _finalize_row(i)
        _maybe_partial(force=True)

        try:
            with pipeline_progress() as progress:
                think_task = None
                if think_pass_todo > 0:
                    think_task = progress.add_task(
                        "Verifier (Thinking Pass)", total=think_pass_todo
                    )
                    by_budget = {}
                    for job in think_jobs:
                        i, r, think_reasons = job
                        budget = self._think_token_budget(task_types[i], think_reasons)
                        by_budget.setdefault(budget, []).append(job)

                    batch_size = max(1, self.think_pass_batch_size)
                    for budget, jobs in sorted(by_budget.items(), key=lambda kv: kv[0]):
                        sized = []
                        for job in jobs:
                            i, r, _reasons = job
                            exemplars = self._retrieve_exemplars(
                                str(r["prompt_bn"]),
                                str(r["response_bn"]),
                                query_emb=query_embs[i],
                            )
                            prompt = self._build_think_prompt(
                                evidence=str(r["context"]),
                                prompt_bn=str(r["prompt_bn"]),
                                response_bn=str(r["response_bn"]),
                                task_type=task_types[i],
                                exemplars=exemplars,
                            )
                            sized.append((len(prompt), prompt, job))
                        sized.sort(key=lambda x: x[0])

                        for start_idx in range(0, len(sized), batch_size):
                            batch = sized[start_idx : start_idx + batch_size]
                            prompts = [p for _n, p, _job in batch]
                            texts = self._batched_think_generate(prompts, budget)
                            for (_n, _p, (i, _r, reasons)), text in zip(batch, texts):
                                _finalize_row(
                                    i,
                                    generated_text=text,
                                    think_max_tokens=budget,
                                    think_reasons=list(reasons),
                                )
                            if think_task is not None:
                                progress.advance(think_task, len(batch))
                            _maybe_partial()

                # Any leftover (shouldn't happen) + cached rows
                for i in range(total_rows):
                    if preds[i] is None:
                        _finalize_row(i)
        finally:
            self.debug_log_path = original_log_path

        _maybe_partial(force=True)

        self.last_nli_debug = nli_debug
        ok(
            f"Verifier done · think {think_count}/{total_rows} "
            f"({100.0 * think_count / max(total_rows, 1):.1f}%) · "
            f"NLI skip-think {nli_skipped_think} · NLI applied {nli_applied_count} · "
            f"cache hits {cache_hits} · think_batch={self.think_pass_batch_size}"
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
