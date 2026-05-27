import os
import json
import uuid
import bm25s
import pickle  # needed?
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import re
from llama_cpp import Llama

# source venv/bin/activate
# ./moulinette/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results --student_answer_path data/output/search_results/dataset_code_public.json --dataset_path datasets_public/public/AnsweredQuestions/dataset_code_public.json
# pip install bm25s langchain_text_splitters langchain_core llama-cpp-python
# export UV_CACHE_DIR=/sgoinfre/gasoares/.cache/uv
# export XDG_CACHE_HOME=/sgoinfre/gasoares/.cache
# export TMPDIR=/sgoinfre/gasoares/tmp

# ./moulinette/scripts/exam_answer.sh --student-path . --moulinette-path ./moulinette/moulinette_pkg/


INDEX_DIR = "data/processed/bm25_index"
CHUNKS_DIR = "data/processed/chunks"
MODEL_PATH = "qwen3-0.6b-q4_k_m.gguf"
os.environ["HF_HOME"] = "/sgoinfre/gasoares/.cache/huggingface"
hf_token = os.getenv("HF_TOKEN")

# ── Pydantic models ──────────────────────────────────────────────────────────


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
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_documents(folder_path: str) -> List[Document]:
    documents: List[Document] = []

    def parse_file(path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as fd:
            return fd.read()

    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.ico', '.pyc')):
                continue
            file_path = os.path.join(root, filename)
            # print(file_path)
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


# ── Chunking ─────────────────────────────────────────────────────────────────

def split_documents(documents: List[Document], chunk_size: int = 2000) -> List[Document]:
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=200,
        add_start_index=True
    )
    md_splitter = RecursiveCharacterTextSplitter(
        language=Language.MARKDOWN,  # if you never run into issues, this is right, if you do, look at this
        chunk_size=chunk_size,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # needed?
    )

    chunks = []
    for doc in documents:
        if doc.metadata["file_path"].endswith(".py"):
            split = py_splitter.split_documents([doc])
        else:
            split = md_splitter.split_documents([doc])
        for chunk in split:
            start = chunk.metadata.get("start_index", 0)
            chunk.metadata["first_character_index"] = start
            chunk.metadata["last_character_index"] = start + len(chunk.page_content)
        chunks.extend(split)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/_-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── BM25 index ───────────────────────────────────────────────────────────────

def build_index(chunks: List[Document]) -> bm25s.BM25:
    corpus = [chunk.page_content for chunk in chunks]
    tokenized = bm25s.tokenize(corpus)
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


# ── Retrieval ────────────────────────────────────────────────────────────────

def doc_to_minimal_source(doc: Document) -> MinimalSource:
    return MinimalSource(
        file_path=doc.metadata["file_path"],
        first_character_index=doc.metadata["first_character_index"],
        last_character_index=doc.metadata["last_character_index"]
    )


def retrieval(query: str, retriever: bm25s.BM25, chunks: List[Document], mode, k: int = 5):  # -> List[MinimalSource]:
    tokenized_query = bm25s.tokenize(normalize_text(query))
    fetch_k = max(k * 2, k + 3)
    results, scores = retriever.retrieve(tokenized_query, k=fetch_k)
    relevant_chunks = [chunks[i] for i in results[0]]
    if mode == "code":
        filtered = [item for item in relevant_chunks if item.metadata["file_path"].endswith(".py")]
    elif mode == "docs":
        filtered = [item for item in relevant_chunks if not item.metadata["file_path"].endswith(".py")]
    filtered = filtered[:k]
    print(f"\nQuery: {query}")
    # print("=== CONTEXT ===")
    # for i, chunk in enumerate(filtered, 1):
    #     print(f"Document {i}: {chunk.metadata['file_path']} "
    #           f"[{chunk.metadata['first_character_index']}:{chunk.metadata['last_character_index']}]")
    sources = [doc_to_minimal_source(chunk) for chunk in filtered]
    context_text = "\n\n".join(
        f"[{chunk.metadata['file_path']}]\n{chunk.page_content}"
        for chunk in filtered
    )
    return sources, context_text  # [doc_to_minimal_source(chunk) for chunk in filtered]


# ── UnasweredQuestions ───────────────────────────────────────────────────────

def unQuestHelper(item) -> UnansweredQuestion:
    return UnansweredQuestion(
        question_id=item["question_id"],
        question=item["question"]
    )


def unQuestOpen(path: str) -> List[UnansweredQuestion]:
    unQuestions = open(path, 'r', encoding="utf-8", errors="ignore")
    values = json.load(unQuestions)
    unQuestions.close()
    return [unQuestHelper(item) for item in values["rag_questions"]]


def unQuestPipeline(path: str, llm: Llama, k=10) -> list[MinimalAnswer]:
    answers: list[MinimalAnswer] = []
    unQuest = unQuestOpen(path)
    if "code" in path:
        mode = "code"
    else:
        mode = "docs"
    for item in unQuest:
        related_sources, context_text = retrieval(
            item.question,
            retriever, chunks, mode, k=k
        )
#         prompt = f"""<|im_start|>system
# Answer using ONLY the provided context in one or two sentences.
# Respond with plain text only.<|im_end|>
# <|im_start|>user
# Context:
# {context_text}
# Question:
# {item.question}<|im_end|>
# <|im_start|>assistant
# """
        prompt = f"""
/No_think
Instructions: Answer using ONLY the context.
Be direct. No preamble.
No markdown. No URLs.
If it is a list use commas.
If not in context: 'Not found in context'.\n\n
Context: The three key abstractions used for disaggregated prefilling
in vLLM are: KV pipe, KV lookup buffer, and KV connector.\n
Question: What are the three key abstractions used for disaggregated
prefilling in vLLM?\n
Answer: KV pipe, KV lookup buffer, and KV connector\n\n
Context: {context_text[:200]}\n
Question: {item.question}\n
Answer:
"""
        output = llm(
            prompt,
            max_tokens=40,
            temperature=0.0,
            echo=False,
            stop=["Question:", "Context:", "Instructions:", "\n\n"],
        )
        answer: str = output["choices"][0]["text"].strip()
        # print("========================================================================================================================")
        # print(prompt)
        # print("########################################################################################################################")
        # print(output)
        # print("########################################################################################################################")
        # print("========================================================================================================================")
        answers.append(
            MinimalAnswer(
                question_id=item.question_id,
                question_str=item.question,
                answer=answer,
                retrieved_sources=related_sources  # ← the List[MinimalSource]
            )
        )
#     answers: list[MinimalAnswer] = []
#     unQuest = unQuestOpen(path)
#     if "code" in path:
#         mode = "code"
#     else:
#         mode = "docs"
#     for item in unQuest:
#         related_chunks = retrieval(
#             item.question,
#             retriever, chunks, mode, k=2
#         )
#         prompt = f"""<|im_start|>system
# Answer using ONLY the provided context in one or two sentences.
# Respond with plain text only.<|im_end|>
# <|im_start|>user /no_think
# Context:
# {related_chunks}
# Question:
# {item.question}<|im_end|>
# <|im_start|>assistant
# """
#         output = llm(
#             prompt,
#             max_tokens=60,
#             temperature=0.0,
#             echo=False,
#             stop=["<|im_end|>", "\n\n"]
#         )
#         answer: str = output["choices"][0]["text"].strip()
#         answers.append(
#             MinimalAnswer(
#                 question_id=item.question_id,
#                 question=item.question,
#                 retrieved_sources=related_chunks,
#                 answer=answer
#             )
#         )
    return answers, mode


def save_answers(answers, mode, withAnswers: bool, k=10):  # list[MinimalAnswer]
    os.makedirs("data/output/search_results", exist_ok=True)
    if withAnswers is True:
        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=k
        )
    else:
        output = StudentSearchResults(
            search_results=answers,
            k=k
        )
    # data = [
    #     {"source": answer.model_dump()}
    #     for answer in answers
    # ]
    with open(f"data/output/search_results/dataset_{mode}_private.json", "w", encoding="utf-8") as fd:
        json.dump(output.model_dump(exclude={"search_results": {"__all__": {"content"}}}), fd, indent=4, ensure_ascii=False)


# def save_answers(answers: StudentSearchResultsAndAnswer, mode):  # list[MinimalAnswer]
#     os.makedirs("data/output/search_results", exist_ok=True)
#     data = [
#         {"source": answer.model_dump()}
#         for answer in answers
#     ]
#     with open(f"data/output/search_results/dataset_{mode}_private.json", "w", encoding="utf-8") as fd:
#         json.dump(data, fd, indent=4, ensure_ascii=False)

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=os.cpu_count() or 8,
        n_gpu_layers=0,
        use_mlock=True,
        verbose=False,
        n_batch=512,
        last_n_tokens_size=0,
        cache_prompt=True
    )
    docs = load_documents("data/raw/vllm-0.10.1")
    chunks = split_documents(docs)
    retriever, chunks = get_or_build_index(chunks)
    # results = retrieval(
    #     "What activation formats does the fused batched MoE layer return in vLLM?",
    #     retriever, chunks, "code", k=10
    # )
    k = 10
    answers, mode = unQuestPipeline('datasets_private/private/UnansweredQuestions/dataset_docs_private.json', llm, k)
    save_answers(answers, mode, True, k)
    # print(answers)
        # print(item.question_str)
    # print(bm25s.tokenize(normalize_text(
    #     "What activation formats does the fused batched MoE layer return in vLLM?"
    # )))
