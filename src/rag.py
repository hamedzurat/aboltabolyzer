import argparse
import glob
import json
import os
import pickle
import tomllib

import numpy as np
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from src.config_utils import apply_runtime_settings, resolve_section
from src.tui import banner, console, done_panel, info, kv_table, ok, pipeline_progress, warn

DEFAULT_RAG_SOURCES = {
    "wiki": {"corpus_dir": "corpus/wiki", "index_path": "indexes/wiki.pkl"},
    "idioms": {"corpus_dir": "corpus/idioms", "index_path": "indexes/idioms.pkl"},
    "literal": {"corpus_dir": "corpus/literal", "index_path": "indexes/literal.pkl"},
    "grammar": {"corpus_dir": "corpus/grammar", "index_path": "indexes/grammar.pkl"},
}

RAG_INDEX_VERSION = 2


def _resolve_numpy_dtype(dtype_name, default=np.float16):
    if dtype_name in (None, "auto"):
        return default
    aliases = {
        "float16": np.float16,
        "fp16": np.float16,
        "float32": np.float32,
        "fp32": np.float32,
    }
    try:
        return aliases[str(dtype_name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported rag.index_dtype: {dtype_name}") from exc


def truncate_evidence(text, max_tokens):
    """Truncate evidence by whitespace tokens (approx) to fit the encoder window."""
    if not text or max_tokens is None or max_tokens <= 0:
        return text
    tokens = str(text).split()
    if len(tokens) <= max_tokens:
        return str(text)
    return " ".join(tokens[:max_tokens])


def resolve_rag_sources(config):
    """Return {source_name: {corpus_dir, index_path}} for typed corpora."""
    rag_config = resolve_section(config, "rag")
    corpus_root = rag_config.get("corpus_root", "corpus")
    index_root = rag_config.get("index_root", "indexes")
    configured = rag_config.get("sources")

    def _paths(name):
        return {
            "corpus_dir": os.path.join(corpus_root, name),
            "index_path": os.path.join(index_root, f"{name}.pkl"),
        }

    # Prefer explicit list: sources = ["wiki", "idioms", ...]
    if isinstance(configured, list) and configured:
        return {str(name): _paths(str(name)) for name in configured}

    # Or explicit table map: [rag.sources.wiki] corpus_dir=... index_path=...
    if isinstance(configured, dict) and configured:
        sources = {}
        for name, spec in configured.items():
            if isinstance(spec, dict):
                sources[name] = {
                    "corpus_dir": spec.get("corpus_dir") or _paths(name)["corpus_dir"],
                    "index_path": spec.get("index_path") or _paths(name)["index_path"],
                }
            else:
                sources[str(name)] = _paths(str(name))
        return sources

    return {name: _paths(name) for name in DEFAULT_RAG_SOURCES}


def source_paths(config, source_name):
    sources = resolve_rag_sources(config)
    if source_name not in sources:
        raise KeyError(f"Unknown RAG source '{source_name}'. Known: {sorted(sources)}")
    return sources[source_name]


class BanglaRAG:
    def __init__(
        self,
        config_path="configs/config.toml",
        config=None,
        source_name=None,
        corpus_dir=None,
        index_path=None,
    ):
        if config is None:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
        self.config = config
        apply_runtime_settings(self.config)

        self.rag_config = resolve_section(self.config, "rag")
        self.source_name = source_name

        if corpus_dir is not None and index_path is not None:
            self.corpus_dir = corpus_dir
            self.index_path = index_path
        elif source_name is not None:
            paths = source_paths(self.config, source_name)
            self.corpus_dir = paths["corpus_dir"]
            self.index_path = paths["index_path"]
        else:
            sources = resolve_rag_sources(self.config)
            self.source_name = "wiki" if "wiki" in sources else next(iter(sources))
            default = sources[self.source_name]
            self.corpus_dir = default["corpus_dir"]
            self.index_path = default["index_path"]

        self.model_name = self.rag_config["model_name"]
        self.top_k = self.rag_config["top_k"]
        self.similarity_threshold = self.rag_config.get("similarity_threshold", 0.5)
        self.max_evidence_tokens = self.rag_config.get("max_evidence_tokens", 512)
        self.batch_size = self.rag_config.get("batch_size", 32)
        self.query_batch_size = self.rag_config.get("query_batch_size", self.batch_size)
        self.max_seq_length = self.rag_config.get("max_seq_length", None)
        self.index_dtype = _resolve_numpy_dtype(self.rag_config.get("index_dtype", "float16"))

        self.model = None
        self.passages = []
        self.embeddings = None
        self.search_embeddings = None

    def load_model(self, force_cpu=False):
        if self.model is None:
            from src.config_utils import resolve_model_path

            resolved_path = resolve_model_path(self.model_name)
            with console.status("Loading SentenceTransformer model...", spinner="aesthetic"):
                device = (
                    "cpu" if force_cpu or os.environ.get("ABOLTABOLYZER_FORCE_CPU") == "1" else None
                )
                if device:
                    self.model = SentenceTransformer(resolved_path, device=device)
                else:
                    self.model = SentenceTransformer(resolved_path)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length
            if self.model.device.type == "cuda":
                self.model = self.model.half()

    def _jsonl_files(self, corpus_dir):
        return sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl")))

    def load_corpus(self):
        passages = []
        jsonl_files = self._jsonl_files(self.corpus_dir)

        # Legacy: typed wiki/ empty but flat corpus/*.jsonl still present.
        if (
            not jsonl_files
            and self.source_name in (None, "wiki")
            and os.path.basename(self.corpus_dir.rstrip("/")) == "wiki"
        ):
            legacy_root = os.path.dirname(self.corpus_dir.rstrip("/")) or "corpus"
            jsonl_files = self._jsonl_files(legacy_root)
            if jsonl_files:
                warn(
                    f"Typed corpus '{self.corpus_dir}' is empty; "
                    f"falling back to flat files in {legacy_root}/"
                )

        if not jsonl_files:
            warn(f"No corpus files (*.jsonl) in '{self.corpus_dir}'")
            return passages

        with pipeline_progress() as progress:
            task = progress.add_task("Reading corpus", total=len(jsonl_files))
            for file_path in jsonl_files:
                progress.update(task, description=f"Reading {os.path.basename(file_path)}")
                skipped = 0
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                if "text" in obj:
                                    passages.append(obj["text"])
                                elif "passage" in obj:
                                    passages.append(obj["passage"])
                            except json.JSONDecodeError:
                                skipped += 1
                                continue
                if skipped:
                    warn(f"Skipped {skipped} unreadable line(s) in {os.path.basename(file_path)}")
                progress.advance(task)

        return passages

    def _compact_embeddings(self, embeddings):
        return np.ascontiguousarray(embeddings, dtype=self.index_dtype)

    def prepare_search_embeddings(self):
        """Return a reusable float32 matrix for fast dot-product search."""
        if self.embeddings is None:
            if not self.load_index():
                return None
        if self.search_embeddings is None:
            self.search_embeddings = np.ascontiguousarray(self.embeddings, dtype=np.float32)
        return self.search_embeddings

    def build_index(self):
        import math

        label = self.source_name or os.path.basename(self.corpus_dir.rstrip("/") or "default")
        info(f"Building '{label}': {self.corpus_dir} → {self.index_path}")
        self.passages = self.load_corpus()

        if not self.passages:
            warn(f"Corpus empty for '{label}' — skip index")
            return False

        total_p = len(self.passages)
        info(f"Loaded {total_p} passages · encoding with batch_size={self.batch_size}")

        self.load_model()

        chunk_size = 5000
        num_chunks = math.ceil(total_p / chunk_size)
        chunks_dir = os.path.join(os.path.dirname(self.index_path) or ".", "chunks", label)
        os.makedirs(chunks_dir, exist_ok=True)

        embeddings_chunks = [None] * num_chunks
        completed_chunks = 0

        for chunk_idx in range(num_chunks):
            chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx}.pkl")
            if os.path.exists(chunk_file):
                try:
                    with open(chunk_file, "rb") as f:
                        chunk_data = pickle.load(f)
                        embeddings_chunks[chunk_idx] = chunk_data["embeddings"]
                        completed_chunks += 1
                except Exception:
                    os.remove(chunk_file)

        if completed_chunks > 0:
            info(f"Resuming: {completed_chunks}/{num_chunks} chunks already encoded")

        batch_size = self.batch_size
        with pipeline_progress() as progress:
            task = progress.add_task(f"Encode '{label}'", total=num_chunks)
            progress.advance(task, completed_chunks)

            for chunk_idx in range(num_chunks):
                if embeddings_chunks[chunk_idx] is not None:
                    continue

                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_p)
                chunk_passages = self.passages[start_idx:end_idx]
                progress.update(
                    task,
                    description=(
                        f"Encode '{label}' · passages {start_idx + 1}–{end_idx}/{total_p}"
                    ),
                )

                chunk_embeddings = self.model.encode(
                    chunk_passages,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                )
                chunk_embeddings = self._compact_embeddings(chunk_embeddings)

                chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx}.pkl")
                with open(chunk_file, "wb") as f:
                    pickle.dump(
                        {
                            "index_version": RAG_INDEX_VERSION,
                            "embedding_dtype": np.dtype(self.index_dtype).name,
                            "embeddings": chunk_embeddings,
                        },
                        f,
                    )

                embeddings_chunks[chunk_idx] = chunk_embeddings
                progress.advance(task, 1)

        info("Merging embedding chunks...")
        self.embeddings = self._compact_embeddings(np.vstack(embeddings_chunks))
        self.search_embeddings = None

        info(f"Saving index → {self.index_path}")
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "index_version": RAG_INDEX_VERSION,
                    "embedding_dtype": np.dtype(self.index_dtype).name,
                    "model_name": self.model_name,
                    "passages": self.passages,
                    "embeddings": self.embeddings,
                },
                f,
            )

        try:
            for chunk_idx in range(num_chunks):
                chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx}.pkl")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            os.rmdir(chunks_dir)
        except Exception as e:
            warn(f"Could not clean up temporary chunks: {e}")

        ok(f"Dense index '{label}' ready ({total_p} passages)")
        return True

    def load_index(self):
        if not os.path.exists(self.index_path):
            warn(f"Index not found: {self.index_path} — run `just make-rag` first")
            return False

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.passages = data["passages"]
            self.embeddings = self._compact_embeddings(data["embeddings"])
            self.search_embeddings = None
        ok(
            f"Loaded index '{self.source_name or self.index_path}' · "
            f"{len(self.passages)} passages · {self.embeddings.dtype}"
        )
        return True

    def retrieve(self, query, top_k=None, similarity_threshold=None):
        """Return list of {"text": str, "score": float} above the similarity threshold."""
        if self.embeddings is None:
            if not self.load_index():
                return []

        self.load_model()

        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        index = self.prepare_search_embeddings()
        if index is None:
            return []

        query_embedding = np.asarray(
            self.model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0],
            dtype=np.float32,
        )

        similarities = np.dot(index, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_indices:
            score = float(similarities[i])
            if score >= similarity_threshold:
                results.append({"text": self.passages[i], "score": score})
        return results

    def format_evidence(self, hits):
        """Join retrieved hits and truncate to max_evidence_tokens."""
        if not hits:
            return "[NULL]", 0, float("nan"), float("nan")

        joined = " ".join(hit["text"] for hit in hits)
        evidence = truncate_evidence(joined, self.max_evidence_tokens)
        scores = [hit["score"] for hit in hits]
        return evidence, len(hits), float(max(scores)), float(np.mean(scores))


def build_all_indexes(config_path="configs/config.toml", sources=None):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    all_sources = resolve_rag_sources(config)
    names = list(sources) if sources else list(all_sources)

    status = {}
    for name in names:
        if name not in all_sources:
            status[name] = "unknown"
            continue
        corpus_dir = all_sources[name]["corpus_dir"]
        n_files = len(glob.glob(os.path.join(corpus_dir, "*.jsonl")))
        status[name] = f"{n_files} jsonl" if n_files else "empty"
    kv_table("Corpus folders", status, key_header="source")

    built = []
    skipped = []
    for i, name in enumerate(names, start=1):
        if name not in all_sources:
            warn(f"[{i}/{len(names)}] Unknown source '{name}' — skip")
            skipped.append(name)
            continue
        info(f"[{i}/{len(names)}] Building source '{name}'")
        rag = BanglaRAG(config=config, source_name=name)
        if rag.build_index():
            built.append(name)
            ok(f"Built indexes/{name}.pkl")
        else:
            skipped.append(name)
            warn(f"Skipped '{name}' (empty or missing corpus)")

    done_panel(
        "Typed RAG indexes",
        [
            f"Built: [bold green]{', '.join(built) if built else 'none'}[/bold green]",
            f"Skipped: [bold yellow]{', '.join(skipped) if skipped else 'none'}[/bold yellow]",
        ],
    )
    return built, skipped


def main():
    parser = argparse.ArgumentParser(description="Bangla RAG Dense Indexer and Retriever")
    parser.add_argument(
        "--make-rag",
        "--build-index",
        dest="make_rag",
        action="store_true",
        help="Build typed dense indexes from corpus/<source>/ folders",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Build/query only this source (repeatable). Default: all sources.",
    )
    parser.add_argument("--query", type=str, help="Query to search")
    args = parser.parse_args()

    if args.make_rag:
        banner("Make RAG indexes", "corpus/<source>/*.jsonl → indexes/<source>.pkl")
        build_all_indexes(sources=args.sources)
    elif args.query:
        source = (args.sources or ["wiki"])[0]
        info(f"Querying source '{source}'")
        rag = BanglaRAG(source_name=source)
        if rag.load_index():
            results = rag.retrieve(args.query)
            console.print(f"\n[bold cyan]Source:[/bold cyan] {source}")
            console.print(f"[bold cyan]Query:[/bold cyan] {args.query}")
            console.print(f"[bold green]Top {len(results)} matches:[/bold green]")
            for idx, res in enumerate(results):
                console.print(
                    Panel(
                        f"[dim]score={res['score']:.4f}[/dim]\n{res['text']}",
                        title=f"Match {idx + 1}",
                        border_style="cyan",
                    )
                )
        else:
            console.print("[bold red]Failed to run retrieval.[/bold red]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
