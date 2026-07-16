import argparse
import glob
import json
import os
import pickle
import tomllib

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sentence_transformers import SentenceTransformer

from src.config_utils import apply_runtime_settings

console = Console()


def truncate_evidence(text, max_tokens):
    """Truncate evidence by whitespace tokens (approx) to fit the encoder window."""
    if not text or max_tokens is None or max_tokens <= 0:
        return text
    tokens = str(text).split()
    if len(tokens) <= max_tokens:
        return str(text)
    return " ".join(tokens[:max_tokens])


class BanglaRAG:
    def __init__(self, config_path="configs/config.toml"):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)
        apply_runtime_settings(self.config)

        from src.config_utils import resolve_section

        self.rag_config = resolve_section(self.config, "rag")

        self.corpus_dir = self.rag_config["corpus_dir"]
        self.index_path = self.rag_config["index_path"]
        self.model_name = self.rag_config["model_name"]
        self.top_k = self.rag_config["top_k"]
        self.similarity_threshold = self.rag_config.get("similarity_threshold", 0.5)
        self.max_evidence_tokens = self.rag_config.get("max_evidence_tokens", 512)
        self.batch_size = self.rag_config.get("batch_size", 32)
        self.query_batch_size = self.rag_config.get("query_batch_size", self.batch_size)
        self.max_seq_length = self.rag_config.get("max_seq_length", None)

        self.model = None
        self.passages = []
        self.embeddings = None

    def load_model(self):
        if self.model is None:
            from src.config_utils import resolve_model_path

            resolved_path = resolve_model_path(self.model_name)
            with Console().status("Loading SentenceTransformer model...", spinner="aesthetic"):
                self.model = SentenceTransformer(resolved_path)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length
            if self.model.device.type == "cuda":
                self.model = self.model.half()

    def load_corpus(self):
        passages = []
        jsonl_files = glob.glob(os.path.join(self.corpus_dir, "*.jsonl"))

        if not jsonl_files:
            console.print(
                f"[bold red]Warning: No corpus files (*.jsonl) found in directory '{self.corpus_dir}'[/bold red]"
            )
            return passages

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            task = progress.add_task(description="Reading corpus files...", total=len(jsonl_files))

            for file_path in jsonl_files:
                progress.update(task, description=f"Reading {os.path.basename(file_path)}...")
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                if "text" in obj:
                                    passages.append(obj["text"])
                                elif "passage" in obj:
                                    passages.append(obj["passage"])
                            except json.JSONDecodeError:
                                continue
                progress.advance(task)

        # Sort passages by length to optimize encoder performance
        passages = sorted(passages, key=len)
        return passages

    def build_index(self):
        console.print("[bold cyan]Step 1:[/bold cyan] Loading corpus passages...")
        self.passages = self.load_corpus()

        if not self.passages:
            console.print("[bold red]Corpus is empty. Cannot build index.[/bold red]")
            return False

        total_p = len(self.passages)
        console.print(f"Loaded {total_p} passages. Computing dense embeddings...")

        self.load_model()

        import math

        chunk_size = 5000
        num_chunks = math.ceil(total_p / chunk_size)
        chunks_dir = os.path.join(os.path.dirname(self.index_path), "chunks")
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
            console.print(
                f"Found [bold green]{completed_chunks}/{num_chunks}[/bold green] already completed chunks. Resuming..."
            )

        batch_size = self.batch_size
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Encoding passages...", total=num_chunks)
            progress.advance(task, completed_chunks)

            for chunk_idx in range(num_chunks):
                if embeddings_chunks[chunk_idx] is not None:
                    continue

                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_p)
                chunk_passages = self.passages[start_idx:end_idx]

                chunk_embeddings = self.model.encode(
                    chunk_passages,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                )

                chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx}.pkl")
                with open(chunk_file, "wb") as f:
                    pickle.dump({"embeddings": chunk_embeddings}, f)

                embeddings_chunks[chunk_idx] = chunk_embeddings
                progress.advance(task, 1)

        console.print("Merging all chunks...")
        self.embeddings = np.vstack(embeddings_chunks)

        console.print(f"Saving index to [italic]{self.index_path}[/italic]...")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"passages": self.passages, "embeddings": self.embeddings}, f)

        # Cleanup chunk files
        try:
            for chunk_idx in range(num_chunks):
                chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_idx}.pkl")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            os.rmdir(chunks_dir)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not clean up temporary chunks: {e}[/yellow]")

        console.print(
            Panel(
                "[bold green]✔ Dense index built and saved successfully![/bold green]",
                border_style="green",
            )
        )
        return True

    def load_index(self):
        if not os.path.exists(self.index_path):
            console.print(
                f"[bold red]Index file {self.index_path} not found. Build the index first.[/bold red]"
            )
            return False

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.passages = data["passages"]
            self.embeddings = data["embeddings"]
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

        query_embedding = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0]

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in top_indices:
            score = float(similarities[i])
            if score >= similarity_threshold:
                results.append({"text": self.passages[i], "score": score})
        return results

    def retrieve_many(self, queries, top_k=None, similarity_threshold=None):
        """Batch-encode queries, then retrieve hits for each query."""
        queries = list(queries)
        if not queries:
            return []
        if self.embeddings is None:
            if not self.load_index():
                return [[] for _ in queries]

        self.load_model()

        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        query_embeddings = self.model.encode(
            queries,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=self.query_batch_size,
        )

        all_results = []
        for query_embedding in query_embeddings:
            similarities = np.dot(self.embeddings, query_embedding)
            if top_k < len(similarities):
                candidate_indices = np.argpartition(similarities, -top_k)[-top_k:]
                top_indices = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
            else:
                top_indices = np.argsort(similarities)[::-1]

            results = []
            for i in top_indices[:top_k]:
                score = float(similarities[i])
                if score >= similarity_threshold:
                    results.append({"text": self.passages[i], "score": score})
            all_results.append(results)
        return all_results

    def format_evidence(self, hits):
        """Join retrieved hits and truncate to max_evidence_tokens."""
        if not hits:
            return "[NULL]", 0, float("nan"), float("nan")

        joined = " ".join(hit["text"] for hit in hits)
        evidence = truncate_evidence(joined, self.max_evidence_tokens)
        scores = [hit["score"] for hit in hits]
        return evidence, len(hits), float(max(scores)), float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="Bangla RAG Dense Indexer and Retriever")
    parser.add_argument(
        "--build-index", action="store_true", help="Build Dense index from corpus directory"
    )
    parser.add_argument("--query", type=str, help="Query to search")
    args = parser.parse_args()

    rag = BanglaRAG()
    if args.build_index:
        console.print(
            Panel("[bold yellow]Dense RAG Indexer Phase[/bold yellow]", border_style="yellow")
        )
        rag.build_index()
    elif args.query:
        if rag.load_index():
            results = rag.retrieve(args.query)
            console.print(f"\n[bold cyan]Query:[/bold cyan] {args.query}")
            console.print(f"[bold green]Top {len(results)} matches retrieved:[/bold green]")
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


if __name__ == "__main__":
    main()
