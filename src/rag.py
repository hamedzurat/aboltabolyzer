import argparse
import glob
import json
import os
import pickle
import re
import tomllib

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sentence_transformers import SentenceTransformer

console = Console()


def bengali_tokenize(text):
    return re.findall(r"[\u0980-\u09ff\w]+", text.lower())


class BanglaRAG:
    def __init__(self, config_path="configs/config.toml"):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.corpus_dir = self.config["rag"]["corpus_dir"]
        self.index_path = self.config["rag"]["index_path"]
        self.model_name = self.config["rag"]["model_name"]
        self.top_k = self.config["rag"]["top_k"]
        self.similarity_threshold = self.config["rag"].get("similarity_threshold", 0.5)

        self.model = None
        self.passages = []
        self.embeddings = None

    def load_model(self):
        if self.model is None:
            from src.config_utils import resolve_model_path

            resolved_path = resolve_model_path(self.model_name)
            with Console().status("Loading SentenceTransformer model...", spinner="aesthetic"):
                self.model = SentenceTransformer(resolved_path)

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

        batch_size = 128
        self.embeddings = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Encoding passages...", total=total_p)
            for i in range(0, total_p, batch_size):
                batch = self.passages[i : i + batch_size]
                batch_embeddings = self.model.encode(
                    batch, show_progress_bar=False, normalize_embeddings=True
                )
                self.embeddings.append(batch_embeddings)
                progress.advance(task, len(batch))

        self.embeddings = np.vstack(self.embeddings)

        console.print(f"Saving index to [italic]{self.index_path}[/italic]...")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"passages": self.passages, "embeddings": self.embeddings}, f)

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
        if self.embeddings is None:
            if not self.load_index():
                return []

        self.load_model()

        if top_k is None:
            top_k = self.top_k
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        # Encode query
        query_embedding = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0]

        # Since embeddings and query_embedding are normalized, cosine similarity is dot product
        similarities = np.dot(self.embeddings, query_embedding)

        # Get sorted indices descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by similarity threshold
        results = [self.passages[i] for i in top_indices if similarities[i] >= similarity_threshold]

        return results


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
                console.print(Panel(res, title=f"Match {idx + 1}", border_style="cyan"))
        else:
            console.print("[bold red]Failed to run retrieval.[/bold red]")


if __name__ == "__main__":
    main()
