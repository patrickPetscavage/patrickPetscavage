import os
import io
import json
import uuid
import base64
import mimetypes
from typing import Optional

# Extensions treated as images for vision indexing and chat attachments
IMAGE_EXTENSIONS = frozenset(
    {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"}
)

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import chromadb
import pdfplumber
from docx import Document
import openpyxl

# Load environment variables
load_dotenv()

# Cached OpenAI client
@st.cache_resource
def get_client(api_key: str):
    return OpenAI(api_key=api_key)

# Persistent Chroma DB client
@st.cache_resource
def get_chroma():
    return chromadb.PersistentClient(path=".chroma_db")

chroma_client = get_chroma()

CHAT_HISTORY_DIR = ".chat_history"
# Stored under CHAT_HISTORY_DIR/images/<scope>/<thread_id>/<file> — JSON holds POSIX-style keys
CHAT_IMAGE_STORE_PREFIX = "images"

# Session cost tracking (USD estimates from reported token usage; update if OpenAI changes rates)
SESSION_COST_STATE_KEY = "api_cost_totals_v1"
# https://openai.com/api/pricing/ — values in USD per 1M tokens
PRICE_GPT4O_MINI_INPUT_PER_M = 0.15
PRICE_GPT4O_MINI_OUTPUT_PER_M = 0.60
PRICE_TEXT_EMBEDDING_3_SMALL_PER_M = 0.02


def fresh_cost_totals() -> dict:
    return {
        "completion": {
            "rag_answer": {"in": 0, "out": 0},
            "vision_index": {"in": 0, "out": 0},
            "vision_memory": {"in": 0, "out": 0},
        },
        "embedding": {"index": 0, "retrieve": 0},
    }


def session_cost_totals() -> dict:
    if SESSION_COST_STATE_KEY not in st.session_state:
        st.session_state[SESSION_COST_STATE_KEY] = fresh_cost_totals()
    return st.session_state[SESSION_COST_STATE_KEY]


def merge_completion_usage(cost_totals: dict, bucket: str, response) -> None:
    u = getattr(response, "usage", None)
    if not u:
        return
    b = cost_totals["completion"].setdefault(bucket, {"in": 0, "out": 0})
    b["in"] += int(getattr(u, "prompt_tokens", 0) or 0)
    b["out"] += int(getattr(u, "completion_tokens", 0) or 0)


def merge_embedding_usage(cost_totals: dict, kind: str, response) -> None:
    u = getattr(response, "usage", None)
    if not u:
        return
    n = int(
        getattr(u, "total_tokens", None)
        or getattr(u, "prompt_tokens", None)
        or 0
    )
    cost_totals["embedding"][kind] = cost_totals["embedding"].get(kind, 0) + n


def completion_bucket_usd(inp: int, out: int) -> float:
    return (
        inp / 1_000_000.0 * PRICE_GPT4O_MINI_INPUT_PER_M
        + out / 1_000_000.0 * PRICE_GPT4O_MINI_OUTPUT_PER_M
    )


def total_session_usd_est(cost_totals: dict) -> float:
    usd = 0.0
    for b in cost_totals["completion"].values():
        usd += completion_bucket_usd(b.get("in", 0), b.get("out", 0))
    emb = cost_totals["embedding"]
    usd += (
        (emb.get("index", 0) + emb.get("retrieve", 0))
        / 1_000_000.0
        * PRICE_TEXT_EMBEDDING_3_SMALL_PER_M
    )
    return usd


def cost_breakdown_select_options(cost_totals: dict) -> list[str]:
    total = total_session_usd_est(cost_totals)
    lines: list[str] = [
        f"Total (session) — ${total:.4f} USD",
    ]
    c = cost_totals["completion"]
    labels = {
        "rag_answer": "Chat answers (gpt-4o-mini, RAG)",
        "vision_index": "Indexing: describe images/diagrams (gpt-4o-mini)",
        "vision_memory": "Chat: image memory summary (gpt-4o-mini)",
    }
    for key, label in labels.items():
        b = c.get(key) or {"in": 0, "out": 0}
        inp, out = b.get("in", 0), b.get("out", 0)
        if inp == 0 and out == 0:
            continue
        u = completion_bucket_usd(inp, out)
        lines.append(f"{label} — ${u:.4f} ({inp:,} in / {out:,} out tok)")
    ei, er = cost_totals["embedding"].get("index", 0), cost_totals["embedding"].get(
        "retrieve", 0
    )
    if ei:
        lines.append(
            f"Embeddings: indexing (text-embedding-3-small) — ${ei / 1_000_000.0 * PRICE_TEXT_EMBEDDING_3_SMALL_PER_M:.4f} ({ei:,} tok)"
        )
    if er:
        lines.append(
            f"Embeddings: search per question (text-embedding-3-small) — ${er / 1_000_000.0 * PRICE_TEXT_EMBEDDING_3_SMALL_PER_M:.4f} ({er:,} tok)"
        )
    if len(lines) == 1:
        lines.append("No API usage recorded yet this session.")
    return lines


def chat_image_key_to_abs(store_key: str) -> str:
    if not store_key:
        return ""
    rel = store_key.replace("/", os.sep)
    return os.path.normpath(os.path.join(CHAT_HISTORY_DIR, rel))


def save_chat_image_to_store(
    scope_key: str,
    thread_id: str,
    original_name: str,
    image_bytes: bytes,
    mime: str,
) -> str:
    """Persist image bytes; return store key (e.g. images/scope/thread/uuid.jpg)."""
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    ):
        m = (mime or "").lower()
        if "png" in m:
            ext = ".png"
        elif "gif" in m:
            ext = ".gif"
        elif "webp" in m:
            ext = ".webp"
        else:
            ext = ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    store_key = f"{CHAT_IMAGE_STORE_PREFIX}/{scope_key}/{thread_id}/{fname}".replace(
        "\\", "/"
    )
    abs_path = chat_image_key_to_abs(store_key)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        f.write(image_bytes)
    return store_key


def load_chat_image_bytes(store_key: str) -> Optional[bytes]:
    if not store_key:
        return None
    path = chat_image_key_to_abs(store_key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def last_user_message_image(
    messages: list,
) -> Optional[tuple[bytes, str]]:
    """Most recent user turn that has a persisted image (for optional re-send to the model)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        key = m.get("image_store_key")
        mime = m.get("image_mime") or "image/jpeg"
        if not key:
            continue
        data = load_chat_image_bytes(key)
        if data:
            return (data, mime)
    return None


def chat_scope_key(company: str, project: str) -> str:
    return f"{normalize(company)}__{normalize(project)}"

def chat_history_path(company: str, project: str) -> str:
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return os.path.join(CHAT_HISTORY_DIR, f"{chat_scope_key(company, project)}.json")

def load_chat_scope_from_disk(company: str, project: str) -> Optional[dict]:
    path = chat_history_path(company, project)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

def save_chat_scope_to_disk(company: str, project: str, state: dict) -> None:
    path = chat_history_path(company, project)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except OSError:
        pass

def ensure_chat_scope_state(company: str, project: str) -> dict:
    sk = chat_scope_key(company, project)
    if "chats_by_scope" not in st.session_state:
        st.session_state.chats_by_scope = {}
    if sk not in st.session_state.chats_by_scope:
        loaded = load_chat_scope_from_disk(company, project)
        if loaded and isinstance(loaded.get("threads"), dict) and loaded["threads"]:
            st.session_state.chats_by_scope[sk] = loaded
            aid = loaded.get("active")
            if aid not in loaded["threads"]:
                st.session_state.chats_by_scope[sk]["active"] = next(iter(loaded["threads"]))
        else:
            tid = str(uuid.uuid4())
            st.session_state.chats_by_scope[sk] = {
                "threads": {tid: {"title": "New chat", "messages": []}},
                "active": tid,
            }
    return st.session_state.chats_by_scope[sk]


def split_user_message_for_display(text: str) -> tuple[str, Optional[str]]:
    """Strip internal [attached image: filename] suffix for UI; return (body, filename)."""
    marker = "\n[attached image:"
    if marker not in text:
        return (text.strip(), None)
    head, tail = text.split(marker, 1)
    fname = tail.strip()
    if fname.endswith("]"):
        fname = fname[:-1].strip()
    return (head.strip() or "(attachment)", fname)


def thread_title_from_message(text: str, max_len: int = 48) -> str:
    body, _fname = split_user_message_for_display(text)
    t = " ".join(body.split())
    if not t:
        return "New chat"
    return (t[: max_len - 1] + "…") if len(t) > max_len else t

# UI reset helper (clears inputs)
def reset_ui():
    keys_to_clear = [
        "chat_input",
        "search_input",
        "temp_input",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Normalize company/project names
def normalize(name: str):
    return name.strip().lower().replace(" ", "_")


# Get or create company collection in Chroma
def get_company_collection(company_id: str):
    safe_name = f"company_{normalize(company_id)}"
    return chroma_client.get_or_create_collection(name=safe_name)

# File readers (PDF, DOCX, Excel, TXT)
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
    return text

def read_docx(file):
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs])

def read_excel(file):
    wb = openpyxl.load_workbook(io.BytesIO(file.read()), data_only=True)
    text = ""

    for sheet in wb.worksheets:
        text += f"\n## {sheet.title}\n"
        for row in sheet.iter_rows(values_only=True):
            vals = [str(v) for v in row if v is not None]
            if vals:
                text += " | ".join(vals) + "\n"

    return text


def guess_image_mime(filename: str, uploaded_type: Optional[str]) -> str:
    if uploaded_type and uploaded_type.startswith("image/"):
        return uploaded_type
    guessed, _ = mimetypes.guess_type(filename)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


def describe_image_for_rag(
    client: OpenAI,
    image_bytes: bytes,
    mime: str,
    filename: str,
    cost_totals: Optional[dict] = None,
) -> str:
    """Turn a diagram or photo into searchable text for embedding (vision)."""
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    instruction = (
        "Describe this image in detail for document search indexing. "
        "If it is a diagram, floor plan, construction drawing, flowchart, chart, or schematic, "
        "explain components, labels, symbols, relationships, and any visible text, dimensions, or numbers. "
        "If it is a photograph of a site or equipment, note what is shown. Be factual and thorough."
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=900,
    )
    if cost_totals is not None:
        merge_completion_usage(cost_totals, "vision_index", res)
    return (res.choices[0].message.content or "").strip()


def brief_image_memory(
    client: OpenAI,
    image_bytes: bytes,
    mime: str,
    cost_totals: Optional[dict] = None,
) -> str:
    """Short vision summary stored with chat history so follow-up questions still have context."""
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    instruction = (
        "In 5–8 sentences, summarize what this image shows for chat memory: "
        "type (photo, chart, diagram, plan…), main subjects, readable labels/titles, "
        "and key numbers or relationships. No preamble."
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=350,
    )
    if cost_totals is not None:
        merge_completion_usage(cost_totals, "vision_memory", res)
    return (res.choices[0].message.content or "").strip()


# Read multiple uploaded files (uses vision for images / diagrams when client is provided)
def read_files(
    files,
    client: Optional[OpenAI] = None,
    cost_totals: Optional[dict] = None,
):
    all_text = []

    for f in files:
        ext = f.name.split(".")[-1].lower()

        if ext in IMAGE_EXTENSIONS:
            raw = f.read()
            if not client:
                all_text.append(
                    f"[IMAGE: {f.name}]\n(Skipped: OpenAI client required to describe images.)"
                )
                continue
            mime = guess_image_mime(f.name, getattr(f, "type", None))
            desc = describe_image_for_rag(
                client, raw, mime, f.name, cost_totals=cost_totals
            )
            all_text.append(f"[IMAGE / DIAGRAM: {f.name}]\n{desc}")
        elif ext == "pdf":
            txt = read_pdf(f)
            all_text.append(f"[DOCUMENT: {f.name}]\n{txt}")
        elif ext == "docx":
            txt = read_docx(f)
            all_text.append(f"[DOCUMENT: {f.name}]\n{txt}")
        elif ext in ["xlsx", "xlsm"]:
            txt = read_excel(f)
            all_text.append(f"[DOCUMENT: {f.name}]\n{txt}")
        else:
            txt = f.read().decode("utf-8", errors="ignore")
            all_text.append(f"[DOCUMENT: {f.name}]\n{txt}")

    return "\n\n".join(all_text)

# Split text into chunks for embedding
def split_text(text, size=400):
    sections = text.split("\n--- PAGE ")
    chunks = []

    for section in sections:
        words = section.split()
        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i+size])
            chunks.append(chunk)

    return chunks

# Create embeddings using OpenAI
def embed(
    client,
    texts,
    cost_totals: Optional[dict] = None,
    embed_kind: str = "retrieve",
):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    if cost_totals is not None:
        merge_embedding_usage(cost_totals, embed_kind, res)
    return [r.embedding for r in res.data]

# Store chunks + embeddings in Chroma
def store_chunks(collection, project, chunks, client, cost_totals: Optional[dict] = None):
    embeddings = embed(client, chunks, cost_totals=cost_totals, embed_kind="index")
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"project": project} for _ in chunks]
    )

# Retrieve relevant chunks from vector DB
def retrieve(
    collection,
    project,
    client,
    query,
    k=8,
    cost_totals: Optional[dict] = None,
):
    query_embedding = embed(client, [query], cost_totals=cost_totals, embed_kind="retrieve")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"project": project}
    )

    return results["documents"][0] if results["documents"] else []

def retrieval_query_for_turn(
    prior_messages: list,
    current_question: str,
    max_prior_user_turns: int = 3,
) -> str:
    """Combine recent user questions so follow-ups embed closer to the right document topics."""
    user_texts = [m["text"] for m in prior_messages if m.get("role") == "user"]
    tail_users = user_texts[-max_prior_user_turns:] if max_prior_user_turns else []
    parts = tail_users + [current_question]
    return "\n".join(p.strip() for p in parts if p and str(p).strip())

def format_conversation_history(prior_messages: list, max_messages: int = 12) -> str:
    """Last N messages (before the current user turn) for pronouns and follow-ups."""
    if not prior_messages:
        return ""
    tail = prior_messages[-max_messages:]
    lines = []
    for m in tail:
        role = m.get("role", "")
        text = (m.get("text") or "").strip()
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        mem = (m.get("image_memory") or "").strip()
        if role == "user" and mem:
            text = (
                f"{text}\n"
                f"(Visual summary from that message — image not re-sent: {mem})"
            )
        lines.append(f"{label}: {text}")
    return "\n".join(lines)

# Generate final answer using retrieved context + optional conversation memory (+ optional image)
def answer(
    client,
    question,
    context,
    conversation_history: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime: Optional[str] = None,
    images: Optional[list[tuple[bytes, str]]] = None,
    cost_totals: Optional[dict] = None,
):
    history_block = ""
    if conversation_history and conversation_history.strip():
        history_block = f"""
RECENT CONVERSATION (for pronouns and follow-ups; summaries of past images appear here when saved):

{conversation_history}

"""

    system_text = (
        "You are a helpful construction and document assistant.\n"
        "When this request includes a user image, analyze it: read charts, diagrams, plans, labels, "
        "and visible text. Answer the user's question using what you see; do not refuse because "
        "the content is visual or a chart.\n"
        "Use the CONTEXT section for facts from indexed project documents. Use the attached image "
        "for anything visible in the image. Combine both when relevant.\n"
        "If there is no image in this request but the user asks about a prior attachment, use any "
        "“Visual summary” lines in the recent conversation, and say they can re-attach the file for more detail.\n"
        "If multiple images are provided, treat them as one context (e.g. new upload plus an earlier diagram)."
    )

    user_text = f"""{history_block}CONTEXT (indexed documents; may be empty):
{context}

USER QUESTION:
{question}"""

    user_content: list = [{"type": "text", "text": user_text}]
    image_parts: list[tuple[bytes, str]] = []
    if images:
        image_parts.extend(images)
    elif image_bytes and image_mime:
        image_parts.append((image_bytes, image_mime))
    for raw, mime in image_parts:
        b64 = base64.standard_b64encode(raw).decode("ascii")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ],
        max_tokens=500,
    )
    if cost_totals is not None:
        merge_completion_usage(cost_totals, "rag_answer", res)
    return (res.choices[0].message.content or "").strip()

# Main Streamlit app
def main():
    st.title("🏢 Construction AI Bot")

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Enter API key")
        return

    client = get_client(api_key)
    cost_totals = session_cost_totals()

    st.sidebar.header("💵 Session cost (estimate)")
    st.sidebar.metric(
        "Total (USD)",
        f"${total_session_usd_est(cost_totals):.4f}",
        help=(
            "Sum of this browser session’s tracked OpenAI calls, using list prices for "
            "gpt-4o-mini and text-embedding-3-small. Your invoice may differ."
        ),
    )
    st.sidebar.selectbox(
        "Breakdown",
        options=cost_breakdown_select_options(cost_totals),
        key="session_cost_breakdown_select",
    )
    st.sidebar.caption(
        f"Rates used: gpt-4o-mini ${PRICE_GPT4O_MINI_INPUT_PER_M}/${PRICE_GPT4O_MINI_OUTPUT_PER_M} per M in/out; "
        f"text-embedding-3-small ${PRICE_TEXT_EMBEDDING_3_SMALL_PER_M}/M. See openai.com/api/pricing."
    )
    if st.sidebar.button("Reset session cost", key="reset_session_cost_btn"):
        st.session_state[SESSION_COST_STATE_KEY] = fresh_cost_totals()
        st.rerun()

    # (Only showing the UPDATED company section — everything else stays the same)

    # COMPANY MANAGEMENT SECTION
    st.sidebar.header("🏢 Company")

    COMPANY_FILE = ".companies.json"

    def load_companies():
        if not os.path.exists(COMPANY_FILE):
            return []
        with open(COMPANY_FILE, "r") as f:
            return json.load(f)

    def save_companies(companies):
        with open(COMPANY_FILE, "w") as f:
            json.dump(sorted(list(set(companies))), f)

    companies = load_companies()

    # Initialize session state
    if "company" not in st.session_state:
        st.session_state["company"] = None

    if "show_company_input" not in st.session_state:
        st.session_state.show_company_input = False

    col1, col2 = st.sidebar.columns([3, 1])

    with col1:
        selected_company = st.selectbox(
            "Select Company",
            companies if companies else ["No companies yet"],
            key="company_select_clean"
    )

    with col2:
        if st.button("➕"):
            st.session_state.show_company_input = True

    # Inline creation
    if st.session_state.get("show_company_input"):
        new_company = st.sidebar.text_input("New company name")

        if st.sidebar.button("Create"):
            if new_company.strip():
                norm = normalize(new_company)
                companies.append(norm)
                save_companies(companies)

                # IMPORTANT FIX: actually create collection immediately
                get_company_collection(norm)

                st.session_state["company"] = norm
                st.session_state.show_company_input = False
                reset_ui()
                st.rerun()

    # Set active company
    if selected_company != "No companies yet":
        if st.session_state.get("company") != selected_company:
            st.session_state["company"] = selected_company

    company = st.session_state["company"]

    # FIX: safety check so UI doesn't crash
    if company:
        st.sidebar.success(f"Active Company: {company}")
    else:
        st.sidebar.warning("No company selected")

    # FIX: prevent crash when no company exists
    if not company:
        st.stop()

    # FIX: collection must be created AFTER company is guaranteed
    collection = get_company_collection(company)


    # PROJECT MANAGEMENT SECTION
    def get_projects():
        data = collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])

        return sorted(list(set(
            m["project"] for m in metas if m and "project" in m
        )))

    st.sidebar.header("📁 Project")

    new_project = st.sidebar.text_input("Create Project", key="project_input")

    if st.sidebar.button("➕ Set Project"):
        if new_project.strip():
            st.session_state["project"] = new_project
            reset_ui()
            st.rerun()

    projects = get_projects()

    selected_project = st.sidebar.selectbox(
        "Select Project",
        projects if projects else ["No projects yet"],
        key="project_select"
    )

    if selected_project != "No projects yet":
        if st.session_state.get("project") != selected_project:
            st.session_state["project"] = selected_project
            reset_ui()
            st.rerun()

    if "project" not in st.session_state:
        st.session_state["project"] = selected_project

    project = st.session_state["project"]
    st.sidebar.success(f"Active Project: {project}")

    # FILE UPLOAD + INDEXING SECTION
    files = st.sidebar.file_uploader(
        "Upload documents & images/diagrams",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "xlsm",
            "txt",
            "png",
            "jpg",
            "jpeg",
            "gif",
            "webp",
            "bmp",
            "tif",
            "tiff",
        ],
        accept_multiple_files=True,
        key=f"uploader_{company}_{project}",
        help="PDFs/DOCX/Excel/TXT are read as text. Images are described with vision and indexed like text.",
    )

    if st.sidebar.button("Index Documents"):
        if not files:
            st.error("Upload files first")
        else:
            text = read_files(files, client, cost_totals)
            chunks = split_text(text)

            store_chunks(collection, project, chunks, client, cost_totals)

            st.success(f"Indexed {len(chunks)} chunks")

    # CHAT THREADS (per company + project): switch, history, disk persistence
    chat_state = ensure_chat_scope_state(company, project)
    threads: dict = chat_state["threads"]
    active_id = chat_state["active"]

    st.sidebar.header("💬 Chats")

    thread_ids = list(threads.keys())
    if active_id not in threads and thread_ids:
        chat_state["active"] = thread_ids[0]
        active_id = chat_state["active"]
        save_chat_scope_to_disk(company, project, chat_state)
        st.rerun()

    def _thread_label(tid: str) -> str:
        t = threads.get(tid, {})
        title = t.get("title") or "Chat"
        n = len(t.get("messages") or [])
        return f"{title} ({n // 2} turns)" if n else title

    col_new, col_del = st.sidebar.columns(2)
    with col_new:
        if st.button("➕ New chat", key=f"new_chat_{chat_scope_key(company, project)}"):
            tid = str(uuid.uuid4())
            threads[tid] = {"title": "New chat", "messages": []}
            chat_state["active"] = tid
            save_chat_scope_to_disk(company, project, chat_state)
            reset_ui()
            st.rerun()
    with col_del:
        if len(thread_ids) > 1 and st.button("🗑️ Delete", key=f"del_chat_{chat_scope_key(company, project)}"):
            del threads[active_id]
            chat_state["active"] = next(iter(threads.keys()))
            save_chat_scope_to_disk(company, project, chat_state)
            reset_ui()
            st.rerun()

    sel_index = thread_ids.index(active_id) if active_id in thread_ids else 0
    picked = st.sidebar.selectbox(
        "Active chat",
        options=thread_ids,
        index=sel_index,
        format_func=_thread_label,
        key=f"chat_pick_{chat_scope_key(company, project)}",
    )
    if picked != active_id:
        chat_state["active"] = picked
        save_chat_scope_to_disk(company, project, chat_state)
        st.rerun()

    active_thread = threads[chat_state["active"]]
    messages = active_thread.setdefault("messages", [])

    if "image_upload_nonce" not in st.session_state:
        st.session_state.image_upload_nonce = 0

    scope_k = chat_scope_key(company, project)
    pin_key = f"pin_chat_image_{scope_k}"

    chat_image = st.sidebar.file_uploader(
        "Attach diagram to next message (optional)",
        type=["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"],
        key=f"chat_attach_{scope_k}_{st.session_state.image_upload_nonce}",
        help=(
            "Images are saved under .chat_history/images/ so they show in the thread after refresh. "
            "Optional: re-send the latest saved image on every message (see checkbox below)."
        ),
    )

    pin_chat_image = st.sidebar.checkbox(
        "Re-send latest saved diagram with every message",
        key=pin_key,
        help=(
            "Includes the most recent image from this chat in each model request until you upload a new one. "
            "Uses vision input tokens on every turn while enabled."
        ),
    )

    for msg in messages:
        role = msg.get("role")
        text = msg.get("text") or ""
        if role == "user":
            body, att_name = split_user_message_for_display(text)
            store_key = msg.get("image_store_key")
            with st.chat_message("user"):
                st.write(body)
                blob = load_chat_image_bytes(store_key) if store_key else None
                if blob:
                    st.image(io.BytesIO(blob), caption=att_name or "Saved attachment")
                elif att_name:
                    st.caption(
                        f"Attached: {att_name} (file not on disk — send again to restore preview)."
                    )
        else:
            st.chat_message(role).write(text)

    # CHAT INPUT + RAG FLOW
    prompt = st.chat_input("Ask something...", key="chat_input")

    if prompt:
        img_bytes: Optional[bytes] = None
        img_mime: Optional[str] = None
        if chat_image is not None:
            img_bytes = chat_image.read()
            img_mime = guess_image_mime(
                chat_image.name, getattr(chat_image, "type", None)
            )

        with st.chat_message("user"):
            st.write(prompt)
            if img_bytes:
                st.image(io.BytesIO(img_bytes), caption=chat_image.name)

        call_images: Optional[list[tuple[bytes, str]]] = None
        if img_bytes and img_mime:
            call_images = [(img_bytes, img_mime)]
        elif pin_chat_image:
            pinned = last_user_message_image(messages)
            if pinned:
                call_images = [pinned]

        rag_query = retrieval_query_for_turn(messages, prompt)
        if img_bytes:
            rag_query = f"{rag_query}\n[user attached an image/diagram]"
        elif call_images:
            rag_query = f"{rag_query}\n[last saved chat image included in model context]"
        context = retrieve(
            collection, project, client, rag_query, k=8, cost_totals=cost_totals
        )
        context_text = "\n\n".join(context)

        history_text = format_conversation_history(messages)
        reply = answer(
            client,
            prompt,
            context_text,
            conversation_history=history_text or None,
            images=call_images,
            cost_totals=cost_totals,
        )

        st.chat_message("assistant").write(reply)

        if active_thread.get("title") == "New chat":
            active_thread["title"] = thread_title_from_message(prompt)

        user_line = prompt
        if img_bytes:
            user_line = f"{prompt}\n[attached image: {chat_image.name}]"
        user_entry: dict = {"role": "user", "text": user_line}
        if img_bytes:
            sk_img = save_chat_image_to_store(
                scope_k,
                chat_state["active"],
                chat_image.name,
                img_bytes,
                img_mime or "image/jpeg",
            )
            user_entry["image_store_key"] = sk_img
            user_entry["image_mime"] = img_mime or "image/jpeg"
            user_entry["image_memory"] = brief_image_memory(
                client, img_bytes, user_entry["image_mime"], cost_totals=cost_totals
            )
        messages.append(user_entry)
        messages.append({"role": "assistant", "text": reply})
        save_chat_scope_to_disk(company, project, chat_state)

        if img_bytes:
            st.session_state.image_upload_nonce += 1
            st.rerun()

#START APP
if __name__ == "__main__":
    main()
