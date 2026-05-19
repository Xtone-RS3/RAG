from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import List
import uuid

# source venv/bin/activate
# export HF_HOME=/sgoinfre/gasoares/.cache/huggingface

# pip install transformers langchain_community langchain_text_splitters langchain_huggingface langchain_chroma pydantic


model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)


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


def load_documents(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError

    documents = []
    skipped = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path}
                ))
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path}
                    ))
                except Exception as e:
                    skipped.append((file_path, str(e)))

    print(f"Loaded {len(documents)} files, skipped {len(skipped)}")
    if skipped:
        for path, err in skipped:
            print(f"  SKIPPED: {path} — {err}")
    return documents


def split_documents(documents, chunk_size=2000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    source_texts = {doc.metadata["source"]: doc.page_content for doc in documents}
    for chunk in chunks:
        source = chunk.metadata["source"]
        original_text = source_texts.get(source, "")
        start = original_text.find(chunk.page_content)
        if start != -1:
            chunk.metadata["first_character_index"] = start
            chunk.metadata["last_character_index"] = start + len(chunk.page_content)
        else:
            # fallback if find fails (shouldn't happen, but safety net)
            chunk.metadata["first_character_index"] = -1
            chunk.metadata["last_character_index"] = -1
    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n=== Chunk {i+1} ===")
    #         print(f"  Source: {chunk.metadata['source']}")
    #         print(f"  Length: {len(chunk.page_content)} characters")
    #         print(f"  Content: \n{chunk.page_content}")
    #         print("=" * 50)
    # if len(chunks) > 5:
    #     print(f"\n... and {len(chunks) - 5} more chunks")
    print("done splitting")
    return chunks


def doc_to_minimal_source(doc) -> MinimalSource:
    return MinimalSource(
        file_path=doc.metadata["source"],
        first_character_index=doc.metadata["first_character_index"],
        last_character_index=doc.metadata["last_character_index"]
    )


# llm = HuggingFaceEndpoint(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     temperature=0.7,
#     max_length=1024,
# )
# model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",  # sentence-transformers/all-MiniLM-L6-v2
    model_kwargs={"device": "cpu"}
)


def create_vector_store(chunks, persist_dir="db/chroma_db", batch_size=500):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            vectorstore.add_documents(batch)
        print(f"Ingested {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    return vectorstore


def get_or_create_vector_store(chunks, persist_dir="db/chroma_db"):
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
    print("Creating new vector store...")
    return create_vector_store(chunks, persist_dir)


def retrieval(query, persist_dir="db/chroma_db") -> list[MinimalSource]:
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    print(f"User query: {query}")
    print("=== CONTEXT ===")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"First Char Index: {doc.metadata['first_character_index']}")
        print(f"Last Char Index: {doc.metadata['last_character_index']}")
        # print(f"Contents:\n{doc.page_content}\n")
    sources = [doc_to_minimal_source(doc) for doc in relevant_docs]
    return sources


if __name__ == "__main__":
    # TODO remember to put all these outside of main after they are good to go
    # for path, subdirs, files in os.walk("vllm-0.10.1"):
    #     for name in files:
    #         file_path = os.path.join(path, name)
    #         loader = file_path
    docs = load_documents("vllm-0.10.1")
    chunks = split_documents(docs)
    vectorstore = get_or_create_vector_store(chunks)
    test = retrieval("What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module?")
    print(test)
    print("=======")
    db = Chroma(persist_directory="db/chroma_db", embedding_function=embeddings)
    results = db.get(where={"source": {"$contains": "%triton_flash_attention%"}})
    print(results["ids"])
    print("=======")
    db = Chroma(persist_directory="db/chroma_db", embedding_function=embeddings)
    sample = db.get(limit=5)
    print(sample["metadatas"])
    print("=======")
    results = db.get(where={"source": {"$contains": "attention/ops"}})
    print(len(results["ids"]))
    print(results["metadatas"][:3])
    print("=======")
    db = Chroma(persist_directory="db/chroma_db", embedding_function=embeddings)
    all_results = db.get()
    sources = [m["source"] for m in all_results["metadatas"]]
    ops_sources = [s for s in sources if "attention/ops" in s]
    print(f"Total chunks in DB: {len(sources)}")
    print(f"Chunks from attention/ops: {len(ops_sources)}")
    print(ops_sources[:5])
    print("=======")
    triton_sources = [s for s in sources if "triton_flash_attention" in s]
    print(f"Chunks from triton_flash_attention: {len(triton_sources)}")
    print(triton_sources)
