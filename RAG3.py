from __future__ import annotations

from uuid import uuid4
from pydantic import BaseModel, Field
from typing import List
import ast
from pathlib import Path
from typing import List
import json
import bm25s
from tqdm import tqdm
import fire


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    chunk_id: str
    file_path: str
    content: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


def chunk_python_file(
    file_path: Path,
    content: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """
    Chunk Python code using AST nodes.
    """

    chunks: List[Chunk] = []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_text_file(file_path, content, max_chunk_size)

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue

        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue

        start_line = node.lineno - 1
        end_line = node.end_lineno

        lines = content.splitlines(keepends=True)

        start_char = sum(len(x) for x in lines[:start_line])
        end_char = sum(len(x) for x in lines[:end_line])

        chunk_text = content[start_char:end_char]

        if len(chunk_text) > max_chunk_size:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{file_path}:{start_char}:{end_char}",
                file_path=str(file_path),
                content=chunk_text,
                first_character_index=start_char,
                last_character_index=end_char,
            )
        )

    return chunks


def chunk_text_file(
    file_path: Path,
    content: str,
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> List[Chunk]:
    """
    Sliding window chunking for docs/text.
    """

    chunks: List[Chunk] = []

    start = 0

    while start < len(content):
        end = min(start + max_chunk_size, len(content))

        chunk_text = content[start:end]

        chunks.append(
            Chunk(
                chunk_id=f"{file_path}:{start}:{end}",
                file_path=str(file_path),
                content=chunk_text,
                first_character_index=start,
                last_character_index=end,
            )
        )

        if end == len(content):
            break

        start += max_chunk_size - overlap

    return chunks


def save_chunks(chunks: List[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [chunk.model_dump() for chunk in chunks],
            f,
            ensure_ascii=False,
        )


def load_chunks(path: Path) -> List[Chunk]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [Chunk(**x) for x in data]


SUPPORTED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
}


def ingest_repository(
    repo_path: Path,
    output_dir: Path,
    max_chunk_size: int = 2000,
) -> None:
    chunks: List[Chunk] = []

    files = [
        p for p in repo_path.rglob("*")
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    ]

    for file_path in tqdm(files, desc="Chunking files"):
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if file_path.suffix == ".py":
            file_chunks = chunk_python_file(
                file_path,
                content,
                max_chunk_size,
            )
        else:
            file_chunks = chunk_text_file(
                file_path,
                content,
                max_chunk_size,
            )

        chunks.extend(file_chunks)

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    save_chunks(chunks, chunks_dir / "chunks.json")

    corpus = [chunk.content for chunk in chunks]

    corpus_tokens = bm25s.tokenize(corpus)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    retriever.save(str(output_dir / "bm25_index"))

    print(f"Indexed {len(chunks)} chunks")


class Retriever:
    def __init__(
        self,
        index_dir: Path,
        chunks_path: Path,
    ) -> None:
        self.retriever = bm25s.BM25.load(str(index_dir))
        self.chunks: List[Chunk] = load_chunks(chunks_path)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[MinimalSource]:
        query_tokens = bm25s.tokenize(query)

        results, scores = self.retriever.retrieve(
            query_tokens,
            k=k,
        )

        indices = results[0]

        sources: List[MinimalSource] = []

        for idx in indices:
            chunk = self.chunks[idx]

            sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

        return sources


class CLI:
    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        output_dir: str = "data/processed",
        max_chunk_size: int = 2000,
    ) -> None:
        ingest_repository(
            Path(repo_path),
            Path(output_dir),
            max_chunk_size=max_chunk_size,
        )

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> None:
        retriever = Retriever(
            Path("data/processed/bm25_index"),
            Path("data/processed/chunks/chunks.json"),
        )

        results = retriever.search(query, k)

        for result in results:
            print(result.model_dump_json(indent=2))


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
