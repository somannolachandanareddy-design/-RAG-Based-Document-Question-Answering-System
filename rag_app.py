from __future__ import annotations
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")
DOCUMENTS_DIR = BASE_DIR / os.getenv("DOCUMENTS_DIR", "documents")
LOGS_DIR = BASE_DIR / os.getenv("LOGS_DIR", "logs")
for _d in (VECTOR_STORE_DIR, DOCUMENTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
FAISS_INDEX_NAME = "faiss_index"
def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(
        LOGS_DIR / f"app_{datetime.now():%Y-%m-%d}.log", encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
logger = get_logger("rag")
def ensure_api_key() -> None:
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Create a .env file with your Gemini key."
        )
def log_query(question: str, num_sources: int, latency_ms: float) -> None:
    get_logger("rag.query").info(
        "QUERY | q=%r | sources=%d | latency_ms=%.1f", question, num_sources, latency_ms
    )

@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    ensure_api_key()
    logger.info("Initializing embeddings model: %s", GEMINI_EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
    )
@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    ensure_api_key()
    logger.info("Initializing chat model: %s", GEMINI_CHAT_MODEL)
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True,
    )

def get_qa_template():
    from langchain_core.prompts import ChatPromptTemplate
    system_prompt = (
        "You are a precise document question-answering assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer is not contained in the context, say clearly that the "
        "document does not contain that information. Be concise, cite facts from "
        "the context, and never invent details."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Conversation so far:\n{history}\n\n"
                "Context from the document(s):\n{context}\n\n"
                "Question: {question}\n\nAnswer:",
            ),
        ]
    )
def generate_answer(question: str, context: str, history: str = "") -> str:
    chain = get_qa_template() | get_llm()
    response = chain.invoke(
        {"question": question, "context": context, "history": history or "(none)"}
    )
    answer = getattr(response, "content", str(response))
    logger.info("Generated answer (%d chars)", len(answer))
    return answer

def save_uploaded_file(file_name: str, file_bytes: bytes) -> Path:
    dest = DOCUMENTS_DIR / file_name
    dest.write_bytes(file_bytes)
    logger.info("Saved uploaded PDF: %s (%d bytes)", dest.name, len(file_bytes))
    return dest
def load_pdf(path: str | Path):
    from langchain_community.document_loaders import PyPDFLoader
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    docs = PyPDFLoader(str(path)).load()
    for d in docs:
        d.metadata["source"] = path.name
    logger.info("Loaded %d page(s) from %s", len(docs), path.name)
    return docs
def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    logger.info("Split %d doc(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks

def _index_path() -> Path:
    return VECTOR_STORE_DIR / FAISS_INDEX_NAME
def index_exists() -> bool:
    return (_index_path() / "index.faiss").exists()
def build_vector_store(chunks):
    from langchain_community.vectorstores import FAISS
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(_index_path()))
    logger.info("Built FAISS store with %d chunk(s)", len(chunks))
    return store
def load_vector_store():
    from langchain_community.vectorstores import FAISS
    if not index_exists():
        raise FileNotFoundError("No FAISS index found. Upload a PDF first.")
    return FAISS.load_local(
        str(_index_path()), get_embeddings(), allow_dangerous_deserialization=True
    )
def add_to_vector_store(chunks):
    if not index_exists():
        return build_vector_store(chunks)
    store = load_vector_store()
    store.add_documents(chunks)
    store.save_local(str(_index_path()))
    logger.info("Added %d chunk(s) to existing FAISS store", len(chunks))
    return store
def clear_vector_store() -> None:
    path = _index_path()
    if path.exists():
        shutil.rmtree(path)
        logger.info("Cleared FAISS store at %s", path)
def similarity_search_with_score(query: str, k: int):
    return load_vector_store().similarity_search_with_score(query, k=k)
def count_vectors() -> int:
    if not index_exists():
        return 0
    return load_vector_store().index.ntotal

@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: Optional[int]
    chunk_id: Optional[int]
    distance: float
    relevance: float
def _to_relevance(distance: float) -> float:
    return round(1.0 / (1.0 + distance), 4)
def retrieve(query: str, k: int = TOP_K) -> List[RetrievedChunk]:
    results = similarity_search_with_score(query, k=k)
    chunks: List[RetrievedChunk] = []
    for doc, distance in results:
        meta = doc.metadata or {}
        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                source=meta.get("source", "unknown"),
                page=meta.get("page"),
                chunk_id=meta.get("chunk_id"),
                distance=float(distance),
                relevance=_to_relevance(float(distance)),
            )
        )
    logger.info("Retrieved %d chunk(s) for query=%r", len(chunks), query)
    return chunks
def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        loc = c.source
        if c.page is not None:
            loc += f", page {c.page + 1}"
        parts.append(f"[Source {i} — {loc}]\n{c.content}")
    return "\n\n---\n\n".join(parts)

@dataclass
class Turn:
    question: str
    answer: str
    sources: list = field(default_factory=list)
class ChatHistory:
    def __init__(self) -> None:
        self.turns: List[Turn] = []
    def __len__(self) -> int:
        return len(self.turns)
    def add(self, question: str, answer: str, sources: list) -> None:
        self.turns.append(Turn(question, answer, sources))
    def clear(self) -> None:
        self.turns = []
    def recent_context(self, n: int = 3) -> str:
        recent = self.turns[-n:]
        return "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in recent)
    def to_markdown(self) -> str:
        lines = ["# Chat History\n"]
        for i, t in enumerate(self.turns, start=1):
            lines.append(f"## Turn {i}\n\n**Q:** {t.question}\n\n**A:** {t.answer}\n")
        return "\n".join(lines)
    def to_json(self) -> str:
        return json.dumps(
            [
                {"question": t.question, "answer": t.answer, "sources": t.sources}
                for t in self.turns
            ],
            indent=2,
        )


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    section[data-testid="stSidebar"] { background-color: #161a23; }
    .source-card { background-color: #1c2230; border: 1px solid #2a3242;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
    .relevance-badge { display:inline-block; background:#1f6feb; color:#fff;
        border-radius:999px; padding:2px 10px; font-size:12px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)
if "history" not in st.session_state:
    st.session_state.history = ChatHistory()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
def process_pdfs(uploaded_files) -> None:
    progress = st.progress(0.0, text="Starting…")
    total = len(uploaded_files)
    for i, uf in enumerate(uploaded_files, start=1):
        if uf.name in st.session_state.processed_files:
            continue
        progress.progress((i - 0.5) / total, text=f"Reading {uf.name}…")
        path = save_uploaded_file(uf.name, uf.getvalue())
        chunks = split_documents(load_pdf(path))
        add_to_vector_store(chunks)
        st.session_state.processed_files.append(uf.name)
        st.session_state.total_chunks += len(chunks)
        progress.progress(i / total, text=f"Indexed {uf.name}")
    progress.empty()


with st.sidebar:
    st.title("📄 RAG Q&A")
    st.caption("Upload PDFs and ask questions about them.")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing. Add it to your .env file.")
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("📥 Process documents", use_container_width=True):
        try:
            with st.spinner("Processing documents…"):
                process_pdfs(uploaded)
            st.success("Documents indexed!")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process PDFs")
            st.error(f"Processing failed: {exc}")
    st.divider()
    st.subheader("📊 Document Stats")
    c1, c2 = st.columns(2)
    c1.metric("Files", len(st.session_state.processed_files))
    c2.metric("Chunks", st.session_state.total_chunks)
    st.metric("Vectors in store", count_vectors())
    if st.session_state.processed_files:
        st.write("**Indexed files:**")
        for f in st.session_state.processed_files:
            st.write(f"• {f}")
    st.divider()
    colA, colB = st.columns(2)
    if colA.button("🧹 Clear chat", use_container_width=True):
        st.session_state.history.clear()
        st.rerun()
    if colB.button("🗑️ Reset index", use_container_width=True):
        clear_vector_store()
        st.session_state.processed_files = []
        st.session_state.total_chunks = 0
        st.session_state.history.clear()
        st.rerun()
    if len(st.session_state.history):
        st.download_button(
            "⬇️ Download chat (MD)",
            data=st.session_state.history.to_markdown(),
            file_name="chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download chat (JSON)",
            data=st.session_state.history.to_json(),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )


st.title("Document Question Answering")
st.caption("Retrieval-Augmented Generation powered by Gemini + FAISS")
for turn in st.session_state.history.turns:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        if turn.sources:
            with st.expander("📚 Source context"):
                for s in turn.sources:
                    page_str = (
                        f" · page {s['page'] + 1}" if s.get("page") is not None else ""
                    )
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<span class='relevance-badge'>relevance "
                        f"{s.get('relevance', 0)}</span> <b>{s.get('source')}</b>"
                        f"{page_str}<br><br>{s.get('content', '')[:600]}…</div>",
                        unsafe_allow_html=True,
                    )
question = st.chat_input("Ask a question about your documents…")
if question:
    if not index_exists():
        st.warning("Please upload and process a PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                start = time.time()
                with st.spinner("Searching documents…"):
                    chunks = retrieve(question, k=TOP_K)
                    context = build_context(chunks)
                with st.spinner("Generating answer…"):
                    history_ctx = st.session_state.history.recent_context()
                    answer = generate_answer(question, context, history_ctx)
                latency_ms = (time.time() - start) * 1000
                log_query(question, len(chunks), latency_ms)
                st.markdown(answer)
                sources = [
                    {
                        "source": c.source,
                        "page": c.page,
                        "relevance": c.relevance,
                        "content": c.content,
                    }
                    for c in chunks
                ]
                with st.expander("📚 Source context"):
                    for c in chunks:
                        page_str = (
                            f" · page {c.page + 1}" if c.page is not None else ""
                        )
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<span class='relevance-badge'>relevance "
                            f"{c.relevance}</span> <b>{c.source}</b>"
                            f"{page_str}<br><br>{c.content[:600]}…</div>",
                            unsafe_allow_html=True,
                        )
                st.caption(f"⏱️ {latency_ms:.0f} ms · {len(chunks)} sources")
                st.session_state.history.add(question, answer, sources)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to answer question")
                st.error(f"Something went wrong: {exc}")
