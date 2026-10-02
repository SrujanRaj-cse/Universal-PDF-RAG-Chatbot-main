import streamlit as st
import os
import tempfile
import time
import logging
import datetime
import shutil
import html
import hashlib
from pathlib import Path
from typing import List
import textwrap

# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message: str, level: str = "INFO"):
    """Log event to file and session state"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
        
    if "app_logs" not in st.session_state:
        st.session_state["app_logs"] = []
    
    # Use HTML entities for emojis in logs to avoid syntax errors
    emoji = '&#128308;' if level=='ERROR' else ('&#9888;' if level=='WARNING' else '&#128313;')
    st.session_state["app_logs"].insert(0, f"{emoji} {log_entry}")
    
    if len(st.session_state["app_logs"]) > 100:
        st.session_state["app_logs"] = st.session_state["app_logs"][:100]

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False

try:
    from langchain_community.llms import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# Configuration
INDEX_DIR = "faiss_index_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 10

# Tried in order; if one fails (rate limit / model retired) the next is used.
# Override without editing code: GROQ_MODELS="model-a,model-b"
GROQ_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_MODELS", "openai/gpt-oss-120b,llama-3.1-8b-instant"
).split(",") if m.strip()]
OPENAI_FALLBACK_MODEL = "gpt-3.5-turbo-instruct"  # completion-style model used by langchain OpenAI()

# Page Configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="\U0001F4D8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg: #0b1020;
        --panel: #131a2e;
        --panel-2: #182139;
        --border: rgba(148, 163, 184, 0.16);
        --text: #e6eaf4;
        --muted: #94a3b8;
        --accent: #6366f1;
        --accent-2: #22d3ee;
        --ok: #34d399;
        --bad: #f87171;
    }
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    .stApp {
        background:
            radial-gradient(900px 500px at 10% -10%, rgba(99,102,241,0.18), transparent 60%),
            radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,0.10), transparent 55%),
            var(--bg);
        color: var(--text);
    }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

    h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.01em; }
    p, li, label, span { color: var(--text); }
    small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    /* Hero */
    .hero { text-align: center; padding: 1.5rem 0 1rem 0; }
    .hero .badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; color: var(--accent-2);
        background: rgba(34,211,238,0.10); border: 1px solid rgba(34,211,238,0.25);
        margin-bottom: 14px;
    }
    .hero h1 {
        font-size: 2.8rem; font-weight: 700; margin: 0 0 8px 0;
        background: linear-gradient(90deg, #c7d2fe 0%, #67e8f9 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: var(--muted); font-size: 1.05rem; margin: 0; }

    /* Steps */
    .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 1.5rem 0 1rem 0; }
    .step {
        background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
        padding: 14px 16px; transition: border-color .2s, transform .2s;
    }
    .step:hover { border-color: rgba(99,102,241,0.5); transform: translateY(-2px); }
    .step .num {
        width: 26px; height: 26px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 700; color: #fff; background: linear-gradient(135deg, var(--accent), #8b5cf6); margin-bottom: 8px;
    }
    .step .t { font-weight: 600; font-size: 0.95rem; }
    .step .d { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }

    /* Cards */
    .card {
        background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.25rem;
    }
    .card h4 { margin: 0 0 6px 0; font-size: 1rem; }
    .card p { color: var(--muted); font-size: 0.9rem; margin: 0; line-height: 1.55; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: var(--muted); border-radius: 10px 10px 0 0;
        padding: 10px 18px; font-weight: 500; border: none;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); }
    .stTabs [aria-selected="true"] { color: #fff !important; background: rgba(99,102,241,0.14) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0e1426; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .side-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; }
    .side-sub { color: var(--muted); font-size: 0.8rem; margin-bottom: 1rem; }
    .kv { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px dashed var(--border); font-size: 0.85rem; }
    .kv:last-child { border-bottom: none; }
    .kv span:first-child { color: var(--muted); }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
    .pill.ok { color: var(--ok); background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.3); }
    .pill.off { color: var(--bad); background: rgba(248,113,113,0.10); border: 1px solid rgba(248,113,113,0.3); }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
        border-radius: 10px; padding: 0.55rem 1.1rem; font-weight: 600; transition: all .2s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent); color: #fff; background: rgba(99,102,241,0.18);
    }
    .stButton > button[kind="primary"], .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--accent), #8b5cf6); border: none; color: #fff;
        box-shadow: 0 6px 18px rgba(99,102,241,0.35);
    }
    .stFormSubmitButton > button:hover { filter: brightness(1.1); }

    /* Uploader */
    [data-testid="stFileUploader"] section {
        background: var(--panel); border: 1.5px dashed rgba(99,102,241,0.45); border-radius: 14px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--accent-2); }

    /* Inputs */
    .stTextArea textarea {
        background: var(--panel) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 12px !important;
    }
    .stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important; }

    /* Chat */
    .msg { display: flex; gap: 12px; margin: 0 0 14px 0; }
    .msg .av {
        flex: 0 0 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 700; color: #fff;
    }
    .msg.user .av { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
    .msg.bot .av { background: linear-gradient(135deg, #06b6d4, #22d3ee); color: #06202a; }
    .msg .bubble {
        flex: 1; padding: 13px 18px; border-radius: 14px; line-height: 1.6; font-size: 0.96rem;
        border: 1px solid var(--border);
    }
    .msg.user .bubble { background: rgba(99,102,241,0.14); border-color: rgba(99,102,241,0.35); }
    .msg.bot .bubble { background: var(--panel); }
    .meta { color: var(--muted); font-size: 0.75rem; margin: -6px 0 14px 48px; }

    /* Expanders */
    [data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--border) !important; border-radius: 12px; }

    /* Empty state */
    .empty { text-align: center; padding: 3rem 1.5rem; background: var(--panel); border: 1px solid var(--border); border-radius: 18px; }
    .empty .ic {
        width: 64px; height: 64px; margin: 0 auto 14px auto; border-radius: 18px; display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(34,211,238,0.2)); border: 1px solid var(--border);
    }
    .empty h3 { margin: 0 0 6px 0; }
    .empty p { color: var(--muted); margin: 0; }

    .app-footer { text-align: center; color: var(--muted); font-size: 0.8rem; padding: 2rem 0 0.5rem 0; border-top: 1px solid var(--border); margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
    <div class="badge">Retrieval-Augmented Generation</div>
    <h1>Document Q&amp;A</h1>
    <p>Upload your PDFs and get answers grounded in your own documents, with page-level citations.</p>
</div>
<div class="steps">
    <div class="step"><div class="num">1</div><div class="t">Upload PDFs</div><div class="d">Drop one or more files below</div></div>
    <div class="step"><div class="num">2</div><div class="t">Index</div><div class="d">Processing takes ~20-30s</div></div>
    <div class="step"><div class="num">3</div><div class="t">Ask</div><div class="d">Type your question in Chat</div></div>
    <div class="step"><div class="num">4</div><div class="t">Verify</div><div class="d">Expand sources to check</div></div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='side-title'>Settings</div><div class='side-sub'>Model and index controls</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="padding: .6rem 1rem;">
        <div class="kv"><span>Embedding</span><span>all-mpnet-base-v2</span></div>
        <div class="kv"><span>Chunk size</span><span>{CHUNK_SIZE}</span></div>
        <div class="kv"><span>Top-K</span><span>{TOP_K}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Check API keys
    has_groq = bool(os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None))
    has_openai = bool(os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None))

    use_groq = st.checkbox("Prefer Groq LLM", value=GROQ_AVAILABLE and has_groq)
    force_rebuild = st.button("Rebuild Index", help="Force rebuild FAISS index", use_container_width=True)

    groq_ok = GROQ_AVAILABLE and has_groq
    openai_ok = OPENAI_AVAILABLE and has_openai
    st.markdown(f"""
    <div class="card" style="padding: .8rem 1rem; margin-top: 14px;">
        <div class="kv"><span>Groq</span><span class="pill {'ok' if groq_ok else 'off'}">{'Connected' if groq_ok else 'Not set'}</span></div>
        <div class="kv"><span>OpenAI</span><span class="pill {'ok' if openai_ok else 'off'}">{'Connected' if openai_ok else 'Not set'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Download Chat History
    if "messages" in st.session_state and st.session_state["messages"]:
        chat_history_text = f"Chat History - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        chat_history_text += "="*50 + "\n\n"
        
        for role, content in st.session_state["messages"]:
            # Simple cleanup of HTML tags for better readability in text file
            clean_content = content.replace("<b>", "**").replace("</b>", "**")
            # Remove the hidden debug chunks for the text file
            if "|||DEBUG_CHUNKS" in clean_content:
                clean_content = clean_content.split("|||DEBUG_CHUNKS")[0]
            if "|||FOOTER" in clean_content:
                clean_content = clean_content.split("|||FOOTER")[0]
                
            chat_history_text += f"[{role.upper()}]:\n{clean_content.strip()}\n\n{'-'*50}\n\n"
            
        st.download_button(
            label="Download chat history",
            data=chat_history_text,
            file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


# Helper Functions
def save_uploaded_files(uploaded_files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(dest_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def load_documents_langchain(dir_path: str) -> List[Document]:
    docs = []
    pdf_files = list(Path(dir_path).glob("*.pdf"))
    
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
    
    for pdf_path in pdf_files:
        try:
            # Try PyMuPDF first (fast for text-based PDFs)
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Check if we got meaningful text
            total_text = "".join([p.page_content for p in pages])
            
            # If very little text extracted, use OCR mode
            if len(total_text.strip()) < 100:
                log_event(f"Low text content detected in {pdf_path.name}, using OCR mode...")
                try:
                    # Use Unstructured with OCR for image-based PDFs
                    loader = UnstructuredPDFLoader(
                        str(pdf_path),
                        mode="elements",
                        strategy="hi_res"  # High resolution for better OCR
                    )
                    pages = loader.load()
                    log_event(f"OCR extraction completed for {pdf_path.name}")
                except Exception as ocr_error:
                    log_event(f"OCR failed for {pdf_path.name}: {ocr_error}", "WARNING")
                    # Continue with whatever we got from PyMuPDF
            
            # Log sample text to verify extraction
            if pages:
                sample_text = pages[0].page_content[:500].replace('\n', ' ')
                log_event(f"Extracted text sample from {pdf_path.name}: {sample_text}...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(pages)
            log_event(f"Created {len(chunks)} chunks from {pdf_path.name}")
            
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk.page_content,
                    metadata={**chunk.metadata, "source": pdf_path.name}
                ))
        except Exception as e:
            st.error(f"Failed to load {pdf_path.name}: {e}")
            log_event(f"Error loading {pdf_path.name}: {e}", "ERROR")
    
    return docs

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def compute_files_hash(uploaded_files) -> str:
    """Hash of file names + bytes, so the FAISS cache is tied to the exact upload."""
    h = hashlib.sha256()
    for f in sorted(uploaded_files, key=lambda x: x.name):
        h.update(f.name.encode("utf-8"))
        h.update(f.getbuffer())
    return h.hexdigest()[:16]

def build_or_load_faiss(docs: List[Document], index_dir: str, force_rebuild: bool = False):
    embeddings = get_embeddings()
    
    if os.path.exists(index_dir) and not force_rebuild:
        try:
            # Safe here: the pickle was written by this app into our own local cache dir.
            db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            st.success("&#9989; Loaded existing FAISS index")
            log_event("Loaded existing FAISS index")
            return db
        except Exception as e:
            st.warning(f"&#9888; Failed to load index: {e}. Rebuilding...")
            log_event(f"Failed to load index: {e}", "WARNING")
            shutil.rmtree(index_dir, ignore_errors=True)
    
    with st.spinner("&#128296; Building FAISS index..."):
        log_event(f"Building FAISS index from {len(docs)} documents")
        db = FAISS.from_documents(docs, embeddings)
        os.makedirs(index_dir, exist_ok=True)
        db.save_local(index_dir)
        st.success("&#9989; FAISS index built and saved")
        log_event("FAISS index built successfully")
    
    return db

GROUNDED_PROMPT = ChatPromptTemplate.from_template("""You are a document question-answering assistant.
Answer ONLY using the context below. Do not use outside knowledge.

Context:
{context}

Question: {input}

Rules:
- If the context contains the answer, answer clearly and concisely.
- If the context only partly answers it, give the supported part and say what is missing.
- If the context does not contain the answer, reply exactly: "I could not find this in the uploaded documents."
- Never invent facts, numbers, names or page numbers.

Answer:""")

REWRITE_PROMPT = ChatPromptTemplate.from_template("""Rewrite the last question as a standalone question, using the chat history only to resolve references like "it", "that" or "the second one". If it is already standalone, return it unchanged. Output only the question.

Chat history:
{history}

Last question: {question}

Standalone question:""")

def answer_text_only(message: str) -> str:
    """Strip the footer/sources/debug markers we append to stored assistant messages."""
    return message.split("|||FOOTER:")[0].split("**&#128218; Sources:**")[0].strip()

def build_history_text(messages, max_turns: int = 3) -> str:
    lines = []
    for role, content in messages[-(max_turns * 2 + 1):-1]:  # exclude the current question
        who = "User" if role == "user" else "Assistant"
        lines.append(f"{who}: {answer_text_only(content)[:500]}")
    return "\n".join(lines)

def rewrite_question(llm, messages, question: str) -> str:
    """Make follow-up questions standalone so retrieval works. Falls back to the original."""
    history = build_history_text(messages)
    if not history:
        return question
    try:
        out = (REWRITE_PROMPT | llm).invoke({"history": history, "question": question})
        text = getattr(out, "content", out)
        text = str(text).strip().strip('"')
        return text or question
    except Exception as e:
        log_event(f"Question rewrite failed, using original: {e}", "WARNING")
        return question

def get_groq_llm_instance(api_key, model_name):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.0,
        max_tokens=4096
    )

# File Upload
st.markdown("### Your documents")
uploaded_files = st.file_uploader(
    "Drag and drop PDF files here",
    accept_multiple_files=True,
    type=["pdf"],
    help="Upload PDF documents to create your knowledge base"
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Info & Tech Stack", "Logs", "Notes"])

NOTES_CONTENT = """
<div class="grid">
    <div class="card"><h4>Groq integration</h4><p>Set <code>GROQ_API_KEY</code> and the app uses Groq inference via <code>langchain_groq</code>. Models are tried in order, then it falls back to OpenAI if configured.</p></div>
    <div class="card"><h4>Index persistence</h4><p>The FAISS index is cached per upload hash under <code>faiss_index_storage/</code>, so re-uploading the same PDFs is instant. Use <b>Rebuild Index</b> to force a refresh.</p></div>
    <div class="card"><h4>Memory and context</h4><p>History lives in session state. Follow-ups are rewritten into standalone questions using the last 3 turns before retrieval.</p></div>
    <div class="card"><h4>Safety</h4><p>The prompt restricts answers to retrieved context at temperature 0. This reduces hallucination but cannot eliminate it, so check the cited pages.</p></div>
</div>
"""

with tab4:
    st.markdown("### Developer notes")
    st.html(NOTES_CONTENT)

with tab2:
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### How to Use")
        st.markdown("""
        1. **Upload PDFs** using the file uploader above
        2. **Wait max 20-30 Sec** for documents to be processed
        3. **Ask questions** in the Chat tab
        4. **Wait max 20-30 Sec** for model gives you answer
        5. **View sources** by expanding the sources section
        
        **Troubleshooting:**
        - If answers are not found, click **"Rebuild Index"** in the sidebar and re-upload files.
        - Check **"Debug Context"** in the chat to see what the model is reading.
        """)
        
        st.markdown("### Features")
        st.markdown("""
        - **Multi-PDF Support** - Upload multiple documents
        - **Semantic Search** - FAISS vector similarity
        - **Multiple LLM Options** - Groq (fast) or OpenAI
        - **Source Citations** - See where answers come from
        - **Persistent Index** - Faster subsequent queries
        - **Auto Fallback** - Switches models if one fails
        """)

    with col_info2:
        st.markdown("### Tech Stack")
        st.markdown("""
        <div class="card">
            <p><strong>Frontend:</strong> Streamlit</p>
            <p><strong>Orchestration:</strong> LangChain</p>
            <p><strong>Vector DB:</strong> FAISS (Facebook AI Similarity Search)</p>
            <p><strong>Embeddings:</strong> HuggingFace (all-mpnet-base-v2)</p>
            <p><strong>LLM:</strong> Groq (Llama 3) with OpenAI fallback</p>
            <p><strong>Language:</strong> Python 3.9+</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### API Keys")
        st.markdown("""
        Set in `.streamlit/secrets.toml`:
        - `GROQ_API_KEY` (Recommended)
        - `OPENAI_API_KEY` (Fallback)
        """)

with tab3:
    st.markdown("### Application Logs")
    
    # Toolbar
    col_tools1, col_tools2, col_tools3 = st.columns([1.5, 1.5, 4])
    with col_tools1:
        if st.button("Refresh logs", use_container_width=True):
            st.rerun()
            
    with col_tools2:
        # Prepare log data for download
        log_content = ""
        try:
            if os.path.exists('app.log'):
                with open('app.log', 'r') as f:
                    log_content = f.read()
        except:
            pass
            
        if not log_content and "app_logs" in st.session_state:
            log_content = "\n".join(st.session_state["app_logs"])
            
        st.download_button(
            label="Download",
            data=log_content,
            file_name=f"rag_logs_{datetime.datetime.now().strftime('%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Interactive Log Viewer
    st.markdown("""
    <style>
        .log-container {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
        }
        .log-entry {
            padding: 4px 8px;
            border-bottom: 1px solid #2d2d2d;
            display: flex;
            align-items: flex-start;
        }
        .log-entry:hover {
            background-color: #2d2d2d;
        }
        .log-icon {
            margin-right: 10px;
            min-width: 20px;
        }
        .log-time {
            color: #888;
            margin-right: 10px;
            font-size: 0.8rem;
            min-width: 150px;
        }
        .log-msg-info { color: #4ec9b0; }
        .log-msg-warn { color: #ce9178; }
        .log-msg-error { color: #f44747; }
    </style>
    """, unsafe_allow_html=True)
    
    # Collect all logs from both sources
    all_logs = []
    
    # Read from app.log file
    if os.path.exists('app.log'):
        try:
            with open('app.log', 'r', encoding='utf-8') as f:
                file_logs = f.readlines()
                all_logs.extend([log.strip() for log in file_logs if log.strip()])
        except Exception as e:
            st.warning(f"Could not read app.log: {e}")
    
    # Add session state logs
    if "app_logs" in st.session_state and st.session_state["app_logs"]:
        all_logs.extend(st.session_state["app_logs"])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_logs = []
    for log in all_logs:
        if log not in seen:
            seen.add(log)
            unique_logs.append(log)
    
    # Reverse to show most recent first
    unique_logs.reverse()
    
    log_html = '<div class="log-container">'
    
    if unique_logs:
        for log_entry in unique_logs:
            # Parse entry
            icon = "🔹"
            msg_class = "log-msg-info"
            
            # Check for log level
            if "ERROR" in log_entry or "🔴" in log_entry or "&#128308;" in log_entry:
                icon = "🔴"
                msg_class = "log-msg-error"
            elif "WARNING" in log_entry or "⚠️" in log_entry or "&#9888;" in log_entry:
                icon = "⚠️"
                msg_class = "log-msg-warn"
            
            # Clean message - remove HTML entities and emojis
            clean_msg = log_entry.replace("&#128308;", "").replace("&#9888;", "").replace("&#128313;", "")
            clean_msg = clean_msg.replace("🔴", "").replace("⚠️", "").replace("🔵", "").strip()
            
            # Extract timestamp (format: YYYY-MM-DD HH:MM:SS,mmm or [HH:MM:SS])
            timestamp = ""
            if " - " in clean_msg:
                # Format from logging module: "2024-12-03 01:35:18,123 - INFO - message"
                parts = clean_msg.split(" - ", 2)
                if len(parts) >= 3:
                    timestamp = parts[0]
                    clean_msg = parts[2].strip()
            elif clean_msg.startswith("[") and "]" in clean_msg:
                # Format from session state: "[HH:MM:SS] message"
                parts = clean_msg.split("]", 1)
                timestamp = parts[0] + "]"
                clean_msg = parts[1].strip() if len(parts) > 1 else ""
            
            log_html += f"""
            <div class="log-entry">
                <span class="log-icon">{icon}</span>
                <span class="log-time">{html.escape(timestamp)}</span>
                <span class="{msg_class}">{html.escape(clean_msg)}</span>
            </div>
            """
    else:
        log_html += '<div style="color: #666; text-align: center; padding: 20px;">No logs available</div>'
    
    log_html += '</div>'
    st.html(log_html)

with tab1:
    if uploaded_files:
        try:
            files_hash = compute_files_hash(uploaded_files)
            needs_processing = (
                st.session_state.get("processed_hash") != files_hash or force_rebuild
            )
            if needs_processing:
                temp_dir = tempfile.mkdtemp(prefix="rag_chatbot_")
                try:
                    log_event(f"Processing {len(uploaded_files)} uploaded files")
                    saved_paths = save_uploaded_files(uploaded_files, temp_dir)
                    st.success(f"&#9989; Uploaded {len(saved_paths)} file(s)")
                    t0 = time.perf_counter()
                    with st.spinner("&#128214; Loading documents..."):
                        docs = load_documents_langchain(temp_dir)
                        if not docs:
                            st.error("&#10060; No documents loaded")
                            st.stop()
                        st.info(f"&#128196; Loaded {len(docs)} document chunks")
                    t_load = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    db = build_or_load_faiss(docs, os.path.join(INDEX_DIR, files_hash), force_rebuild)
                    t_index = time.perf_counter() - t1
                    log_event(f"Ingestion: load+chunk={t_load:.2f}s, index={t_index:.2f}s, chunks={len(docs)}")
                    st.session_state["ingest_stats"] = {"chunks": len(docs), "load_s": t_load, "index_s": t_index}
                    st.session_state["retriever"] = db.as_retriever(search_kwargs={"k": TOP_K})
                    st.session_state["processed_hash"] = files_hash
                    st.session_state["messages"] = []  # new documents -> fresh conversation
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            if "retriever" in st.session_state:
                retriever = st.session_state["retriever"]
                # Get API keys
                groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
                openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
                prompt = GROUNDED_PROMPT
                st.markdown("### Chat with your documents")
                stats = st.session_state.get("ingest_stats")
                if stats:
                    st.caption(f"Indexed {stats['chunks']} chunks | load+chunk {stats['load_s']:.1f}s | embed+index {stats['index_s']:.1f}s")
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []
                # Display chat history
                for role, message in st.session_state["messages"]:
                    if role == "user":
                        st.markdown(f"""
                        <div class="msg user"><div class="av">You</div><div class="bubble">{html.escape(message)}</div></div>
                        """, unsafe_allow_html=True)
                    else:
                        # Extract debug chunks
                        debug_chunks = []
                        if "|||DEBUG_CHUNKS:" in message:
                            parts_debug = message.split("|||DEBUG_CHUNKS:")
                            message = parts_debug[0]
                            debug_info = parts_debug[1]
                            # Parse chunks
                            if "|||CHUNK" in debug_info:
                                chunk_parts = debug_info.split("|||CHUNK")
                                for chunk in chunk_parts:
                                    if ":" in chunk:
                                        chunk_num, content = chunk.split(":", 1)
                                        debug_chunks.append(f"**Chunk {chunk_num}**: {content.strip()}")

                        # Extract footer and sources
                        footer_text = ""
                        if "|||FOOTER:" in message:
                            parts_footer = message.split("|||FOOTER:")
                            message = parts_footer[0]
                            footer_text = parts_footer[1]
                        
                        parts = message.split("**&#128218; Sources:**")
                        main_answer = parts[0].strip()
                        
                        st.markdown(f"""
                        <div class="msg bot"><div class="av">AI</div><div class="bubble">{html.escape(main_answer).replace(chr(10), '<br>')}</div></div>
                        """, unsafe_allow_html=True)

                        if len(parts) > 1:
                            sources_raw = parts[1].strip().split("\n")
                            with st.expander("View sources", expanded=False):
                                for line in sources_raw:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if line.startswith("**[") and "]**" in line:
                                        st.markdown(f"**{line.replace('**', '')}**")
                                    elif line.startswith(">"):
                                        st.markdown(f"> {line[1:].strip()}")
                        
                        if debug_chunks:
                            with st.expander("Debug Context", expanded=False):
                                for chunk in debug_chunks:
                                    st.markdown(chunk)
                                    st.markdown("---")

                        if footer_text:
                            st.markdown(f"<div class='meta'>{html.escape(footer_text)}</div>", unsafe_allow_html=True)
                
                # Chat input
                # Chat input area
                def submit_query():
                    if st.session_state.get("query_input"):
                        st.session_state["messages"].append(("user", st.session_state["query_input"]))
                        st.session_state["query_input"] = ""

                st.text_area("Your question", key="query_input", placeholder="e.g. Summarise the key findings in chapter 2", height=110)
                
                # Chat controls
                col_actions1, col_actions2 = st.columns(2)
                with col_actions1:
                    st.button("Ask", on_click=submit_query, use_container_width=True, type="primary")
                
                with col_actions2:
                    if st.button("Clear conversation", use_container_width=True):
                        st.session_state["messages"] = []
                        st.rerun()
                
                # Process latest message
                if st.session_state["messages"] and st.session_state["messages"][-1][0] == "user":
                    with st.spinner("Thinking..."):
                        latest_query = st.session_state["messages"][-1][1]
                        log_event(f"Processing query: {latest_query[:50]}...")
                        success = False
                        models_to_try = []
                        all_errors = []
                        if use_groq and GROQ_AVAILABLE and groq_key:
                            models_to_try.extend([(name, "groq") for name in GROQ_MODELS])
                        if OPENAI_AVAILABLE and openai_key:
                            models_to_try.append((OPENAI_FALLBACK_MODEL, "openai"))
                        if not models_to_try:
                            st.error("&#10060; No LLM configured. Please set API keys.")
                            st.stop()
                        for model_name, provider in models_to_try:
                            try:
                                if provider == "groq":
                                    current_llm = get_groq_llm_instance(groq_key, model_name)
                                else:
                                    current_llm = OpenAI(model_name=model_name, temperature=0.0, openai_api_key=openai_key)
                                t_start = time.perf_counter()
                                standalone_q = rewrite_question(current_llm, st.session_state["messages"], latest_query)
                                t_rewrite = time.perf_counter() - t_start

                                # Retrieval (timed separately from generation)
                                t_r0 = time.perf_counter()
                                source_docs = retriever.invoke(standalone_q)
                                t_retrieve = time.perf_counter() - t_r0

                                # Generation
                                t_g0 = time.perf_counter()
                                if source_docs:
                                    document_chain = create_stuff_documents_chain(current_llm, prompt)
                                    answer = document_chain.invoke({"input": standalone_q, "context": source_docs})
                                else:
                                    answer = "I could not find this in the uploaded documents."
                                t_generate = time.perf_counter() - t_g0
                                answer = str(getattr(answer, "content", answer))
                                t_total = time.perf_counter() - t_start

                                log_event(
                                    f"Latency [{model_name}] rewrite={t_rewrite:.2f}s retrieve={t_retrieve:.2f}s "
                                    f"generate={t_generate:.2f}s total={t_total:.2f}s | chunks={len(source_docs)}"
                                )
                                full_answer = answer
                                full_answer += (
                                    f"|||FOOTER:Model: {model_name} | retrieved {len(source_docs)} chunks | "
                                    f"retrieval {t_retrieve*1000:.0f} ms | generation {t_generate:.2f} s | total {t_total:.2f} s"
                                )
                                if source_docs:
                                    sources_text = "\n\n**&#128218; Sources:**\n"
                                    for i, doc in enumerate(source_docs, 1):
                                        source = doc.metadata.get("source", f"Document {i}")
                                        page = doc.metadata.get("page")
                                        page_label = f", page {int(page) + 1}" if isinstance(page, (int, float)) else ""
                                        raw_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                        snippet = " ".join(raw_snippet.split())
                                        sources_text += f"\n**[{i}]** {source}{page_label}\n> {snippet}\n"
                                    full_answer += sources_text
                                    full_answer += f"\n\n|||DEBUG_CHUNKS:{len(source_docs)}"
                                    for i, doc in enumerate(source_docs, 1):
                                        chunk_preview = doc.page_content[:500].replace('\n', ' ')
                                        full_answer += f"\n|||CHUNK{i}:{chunk_preview}"
                                else:
                                    log_event("WARNING: No source documents retrieved!", "WARNING")
                                st.session_state["messages"].append(("assistant", full_answer))
                                success = True
                                break
                            except Exception as e:
                                error_msg = str(e)
                                all_errors.append(f"{model_name}: {error_msg}")
                                st.warning(f"&#9888; {model_name} failed: {error_msg[:100]}...")
                                log_event(f"Model {model_name} failed: {error_msg}", "WARNING")
                                continue
                        if not success:
                            st.error("&#10060; All models failed. Please check your API keys or try again later.")
                            with st.expander("&#128269; View Error Details"):
                                for error in all_errors:
                                    st.code(error, language="text")
                            st.session_state["messages"].append(("assistant", "&#10060; Sorry, I couldn't generate an answer."))
                    st.rerun()
                # Clear chat button

        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            log_event(f"App error: {e}", "ERROR")

    else:
        st.markdown("""
        <div class="empty">
            <div class="ic">&#128196;</div>
            <h3>Upload a PDF to get started</h3>
            <p>Use the uploader above. Multiple files are supported (up to 200MB each).</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">Document Q&amp;A &bull; Built with LangChain, FAISS &amp; Streamlit</div>
""", unsafe_allow_html=True)