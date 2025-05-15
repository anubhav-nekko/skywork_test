# app.py

import os
import json
import torch
import torchvision.transforms as T
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_DIR    = "/home/ubuntu/models/Skywork-R1V2-38B"
UPLOADS_DIR  = "uploads"
META_FILE    = "metadata.json"
CHAT_MAX_TOK = 1024

# ─── HELPERS ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        device_map="balanced_low_0",
        torch_dtype="auto"
    ).eval()
    return tok, model

def pdf_page_to_tensor_fullres(path: str):
    """
    Render each page at the PDF's native resolution,
    convert to a [1,3,H,W] bfloat16 CUDA tensor.
    """
    doc = fitz.open(path)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        tensor = (
            transform(img)
            .unsqueeze(0)
            .to(torch.bfloat16)
            .cuda()
        )
        pages.append(tensor)
    return pages

def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {"chats": []}

def save_metadata(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

# ─── APP SETUP ────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide")
tok, model = load_model_and_tokenizer()
metadata   = load_metadata()

# Sidebar: Past conversations
st.sidebar.header("📂 Past Conversations")
for idx, chat in enumerate(metadata["chats"]):
    title = chat.get("title") or f"Chat #{idx+1}"
    if st.sidebar.button(title):
        st.session_state.current = chat.copy()
        st.session_state.idx = idx

if "current" not in st.session_state:
    st.session_state.current = {"title": "", "files": [], "history": []}
    st.session_state.idx = None

# Top‐level tabs
tab_chat, tab_upload = st.tabs(["Chat", "Upload PDFs"])

# ─── UPLOAD TAB ───────────────────────────────────────────────────────────────
with tab_upload:
    st.title("Upload PDF Documents")
    uploaded = st.file_uploader("Choose PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        for f in uploaded:
            path = os.path.join(UPLOADS_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s).")
    if st.button("Go to Chat"):
        st.experimental_rerun()

# ─── CHAT TAB ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.title("📄 PDF-Powered Chat")

    # File selection
    files = sorted(os.listdir(UPLOADS_DIR)) if os.path.isdir(UPLOADS_DIR) else []
    chosen = st.multiselect(
        "Select document(s) to include:",
        files,
        default=st.session_state.current["files"]
    )
    st.session_state.current["files"] = chosen

    # Display history
    for msg in st.session_state.current["history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Assistant:** {msg['text']}")

    # New question
    question = st.text_input("Ask a question:")
    if st.button("Send") and question:
        # 1️⃣ Record user message
        st.session_state.current["history"].append({"role":"user","text":question})
        st.markdown(f"**You:** {question}")

        # 2️⃣ Build pixel_values (full-res pages)
        tensors = []
        for fn in chosen:
            path = os.path.join(UPLOADS_DIR, fn)
            tensors.extend(pdf_page_to_tensor_fullres(path))
        pixel_values = torch.cat(tensors, dim=0) if tensors else None

        # 3️⃣ Inference via chat()
        resp = model.chat(
            tok,
            pixel_values,
            question,
            generation_config={"max_new_tokens": CHAT_MAX_TOK}
        )
        assistant = resp["assistant"] if isinstance(resp, dict) else resp

        # 4️⃣ Record & display assistant
        st.session_state.current["history"].append({"role":"assistant","text":assistant})
        st.markdown(f"**Assistant:** {assistant}")

        # 5️⃣ Persist to metadata.json
        if st.session_state.idx is None:
            metadata["chats"].append(st.session_state.current)
            st.session_state.idx = len(metadata["chats"]) - 1
        else:
            metadata["chats"][st.session_state.idx] = st.session_state.current
        save_metadata(metadata)
