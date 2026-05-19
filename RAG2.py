import os
import json
import uuid
import bm25s
import pickle
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import re

# source venv/bin/activate
# pip install bm25s langchain_text_splitters langchain_core

INDEX_DIR = "data/processed/bm25_index"
CHUNKS_DIR = "data/processed/chunks"
os.environ["HF_HOME"] = "/sgoinfre/gasoares/.cache/huggingface"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ── Pydantic models ────────────────────────────────────────────────────────────


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]


# ── Loading ────────────────────────────────────────────────────────────────────

def load_documents(folder_path: str) -> List[Document]:
    documents: List[Document] = []

    def parse_file(path: str) -> str:
        with open(path, "r") as fd:  # , encoding="utf-8", errors="ignore"
            return fd.read()

    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.ico', '.pyc')):
                continue

            file_path = os.path.join(root, filename)

            print(file_path)

            content = parse_file(file_path)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "file_path": file_path
                    }
                )
            )

    print(f"Loaded {len(documents)} documents")
    return documents


    # if not os.path.exists(folder_path):
    #     raise FileNotFoundError(f"Folder not found: {folder_path}")

    # documents = []
    # skipped = []

    # for root, dirs, files in os.walk(folder_path):
    #     for filename in files:
    #         if not (filename.endswith(".py") or filename.endswith(".md")):
    #             continue
    #         file_path = os.path.join(root, filename)
    #         try:
    #             with open(file_path, "r", encoding="utf-8") as f:
    #                 content = f.read()
    #         except UnicodeDecodeError:
    #             try:
    #                 with open(file_path, "r", encoding="latin-1") as f:
    #                     content = f.read()
    #             except Exception as e:
    #                 skipped.append((file_path, str(e)))
    #                 continue

    #         documents.append(Document(
    #             page_content=f"{content}",
    #             metadata={"file_path": file_path}
    #         ))

    # print(f"Loaded {len(documents)} files, skipped {len(skipped)}")
    # return documents


# ── Chunking ───────────────────────────────────────────────────────────────────

def split_documents(documents: List[Document], chunk_size: int = 2000) -> List[Document]:
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=200,
        add_start_index=True
    )
    md_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    chunks = []
    for doc in documents:
        if doc.metadata["file_path"].endswith(".py"):
            split = py_splitter.split_documents([doc])
        else:
            split = md_splitter.split_documents([doc])
        # res, pos = {}, 0
        # for i, chunk in enumerate(splitter):
        #     # start = doc.metadata['first_character_index']  # splitter.find(chunk, pos)
        #     start = chunk.metadata.get("start_index", 0)
        #     if start == -1:
        #         continue

        #     end = start + len(chunk.page_content)
        #     pos = end
        #     chunk_id = f"{doc.metadata['file_path']}:{start}:{end}:{i}"

        #     res[chunk_id] = {
        #         "text": chunk,
        #         "file_path": MinimalSource(
        #             file_path=doc.metadata['file_path'],
        #             first_character_index=start,
        #             last_character_index=end
        #         )
        #     }
        for chunk in split:
            start = chunk.metadata.get("start_index", 0)
            chunk.metadata["first_character_index"] = start
            chunk.metadata["last_character_index"] = start + len(chunk.page_content)
            # chunk.page_content = f"# file: {chunk.metadata['file_path']}\n{chunk.page_content}"
        chunks.extend(split)
    print("====================================")
    print(chunks[0])
    print("====================================")
    print(f"Split into {len(chunks)} chunks")
    return chunks


# corpus support


def clean_text(text: str) -> str:
    # lowercase
    text = text.lower()

    # remove markdown tables
    text = re.sub(r'^\|.*\|$', ' ', text, flags=re.MULTILINE)

    # remove markdown formatting
    text = re.sub(r'[`*_>#-]', ' ', text)

    # remove issue/pr references
    text = re.sub(r'gh-(issue|pr):\d+', ' ', text)

    # remove html tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # split snake_case
    text = re.sub(r"_", " ", text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    return text.strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/_-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── BM25 index ─────────────────────────────────────────────────────────────────

def build_index(chunks: List[Document]) -> bm25s.BM25:
    corpus = [chunk.page_content for chunk in chunks]
    # corpus = [v["text"] for v in chunks_data.values()]
    # corpus = [clean_text(chunk.page_content) for chunk in chunks]
    tokenized = bm25s.tokenize(corpus)  # , stopwords=None, stemmer=None
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    return retriever


def save_chunks_json(chunks):
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    path = os.path.join(CHUNKS_DIR, "chunks.json")

    serializable = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with open(path, "w") as f:
        json.dump(serializable, f, indent=4)


def save_index(retriever: bm25s.BM25):
    print("lol")
    os.makedirs(INDEX_DIR, exist_ok=True)
    retriever.save(INDEX_DIR)


def load_chunks_json(path=os.path.join(CHUNKS_DIR, "chunks.json")):
    with open(path) as f:
        data = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in data
    ]
# def save_index(retriever: bm25s.BM25, chunks: List[Document]) -> None:
#     os.makedirs(INDEX_DIR, exist_ok=True)
#     os.makedirs(CHUNKS_DIR, exist_ok=True)
#     retriever.save(INDEX_DIR)
#     # json.dump(chunks, f, indent=2)
#     with open(os.path.join(CHUNKS_DIR, "chunks.json"), "wb") as f:
#         # pickle.dump(chunks, f)
#         json.dump(chunks, f, indent=2)
#     print(f"Index saved to {INDEX_DIR}")


# def load_index() -> tuple[bm25s.BM25, List[Document]]:
#     retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=False)
#     with open(os.path.join(CHUNKS_DIR, "chunks.json"), "rb") as f:
#         chunks = json.load(f)
#     print("Index loaded")
#     return retriever, chunks

def get_or_build_index(chunks):
    chunks_path = os.path.join(CHUNKS_DIR, "chunks.json")

    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("Loading existing index...")

        retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=False)
        chunks = load_chunks_json(chunks_path)

        return retriever, chunks

    print("Building new index...")

    retriever = build_index(chunks)

    save_index(retriever)
    save_chunks_json(chunks)

    return retriever, chunks
# def get_or_build_index(chunks: List[Document]) -> tuple[bm25s.BM25, List[Document]]:
#     if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
#         print("Loading existing index...")
#         return load_chunks_json()
#     print("Building new index...")
#     retriever = build_index(chunks)
#     save_chunks_json(chunks, INDEX_DIR)
#     return retriever, chunks


# ── Retrieval ──────────────────────────────────────────────────────────────────

def doc_to_minimal_source(doc: Document) -> MinimalSource:
    return MinimalSource(
        file_path=doc.metadata["file_path"],
        first_character_index=doc.metadata["first_character_index"],
        last_character_index=doc.metadata["last_character_index"]
    )


def overlap_score(
    question: str,
    content: str
) -> float:
    q_words = set(
        normalize_text(question).split()
    )
    c_words = set(
        normalize_text(content[:500]).split()
    )
    if not q_words:
        return 0.0
    overlap = len(q_words & c_words)
    # normalize by query size
    return overlap / len(q_words)


def retrieval(query: str, retriever: bm25s.BM25, chunks: List[Document], k: int = 15) -> List[MinimalSource]:
    tokenized_query = bm25s.tokenize(normalize_text(query))
    results, scores = retriever.retrieve(tokenized_query, k=k)

    relevant_chunks = [chunks[i] for i in results[0]]
    # candidates = [
    #     self.indexer.metadata[i]
    #     for i in res[0]
    # ]

    ranked = sorted(
        relevant_chunks,
        key=lambda x: overlap_score(
            query,
            x.page_content
        ),
        reverse=True
    )
    # print(ranked[:k])
    return ranked[:k]

    print(f"Query: {query}")
    print("=== CONTEXT ===")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"Document {i}: {chunk.metadata['file_path']} "
              f"[{chunk.metadata['first_character_index']}:{chunk.metadata['last_character_index']}]")

    return [doc_to_minimal_source(chunk) for chunk in relevant_chunks]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents("data/raw/vllm-0.10.1")
    chunks = split_documents(docs)
    retriever, chunks = get_or_build_index(chunks)
    results = retrieval(
        "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?",
        retriever, chunks, k=15
    )
    print(results)
    print(bm25s.tokenize(normalize_text(
        "What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?"
    )))
    # sample = chunks[123].page_content
    # print(sample)
    # print(bm25s.tokenize([sample]))

    # target_chunks = [c for c in chunks if "triton_flash_attention" in c.metadata["file_path"] 
    #                  and "ops" in c.metadata["file_path"]]
    # for c in target_chunks:
    #     if "FP8_MIN" in c.page_content or "FP8_MAX" in c.page_content:
    #         print(c.metadata)
    #         print(c.page_content[:300])
    #         print("---")
