import os
import json
import uuid
import fitz                # PyMuPDF
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STO_DIR    = BASE_DIR / "storage"
FILE_DIR   = STO_DIR / "files"
CHAT_DIR   = STO_DIR / "chats"
META_PATH  = STO_DIR / "meta.json"
MODEL_DIR  = "/home/ubuntu/models/Skywork-R1V2-38B"

for p in (FILE_DIR, CHAT_DIR):
    p.mkdir(parents=True, exist_ok=True)
STO_DIR.mkdir(exist_ok=True)

# ─── Metadata helpers ────────────────────────────────────────────────────────
def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"chat_ids": []}

def save_meta(meta):
    META_PATH.write_text(json.dumps(meta))

meta = load_meta()

def new_chat_id() -> str:
    return uuid.uuid4().hex[:8]

def load_chat(cid: str):
    p = CHAT_DIR / f"{cid}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"title": None, "messages": [], "file_ids": []}

def save_chat(cid: str, data: dict):
    (CHAT_DIR / f"{cid}.json").write_text(json.dumps(data, indent=2))

# ─── Model loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        device_map="balanced_low_0",
        torch_dtype=torch.float16,
    ).eval()
    return tok, model

tok, model = load_model()

# ─── Vision transform ─────────────────────────────────────────────────────────
VISION_TX = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

@st.cache_data
def pdf_to_tensor_bf16(pdf_path: str):
    """Render each page full-res, convert → [1,3,H,W] bfloat16 CUDA tensor."""
    doc = fitz.open(pdf_path)
    tensors = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0,1.0))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        t = VISION_TX(img).unsqueeze(0).to(torch.bfloat16).cuda()
        tensors.append(t)
    return torch.cat(tensors, dim=0)  # [N,3,H,W]

# ─── Session state & sidebar ─────────────────────────────────────────────────
if "chat_id" not in st.session_state:
    st.session_state.chat_id = meta["chat_ids"][0] if meta["chat_ids"] else None

with st.sidebar:
    st.title("📄 PDF Chat")
    # Upload new PDFs
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            fid = uuid.uuid4().hex[:8]
            dest = FILE_DIR / f"{fid}_{f.name}"
            dest.write_bytes(f.read())
            if st.session_state.chat_id:
                cd = load_chat(st.session_state.chat_id)
                cd["file_ids"].append(fid)
                save_chat(st.session_state.chat_id, cd)
        st.rerun()

    st.markdown("### Previous Chats")
    for cid in meta["chat_ids"]:
        title = load_chat(cid)["title"] or cid
        if st.button(title, key=cid):
            st.session_state.chat_id = cid

    if st.button("➕ New chat"):
        new_id = new_chat_id()
        meta["chat_ids"].insert(0, new_id)
        save_meta(meta)
        save_chat(new_id, {"title": None, "messages": [], "file_ids": []})
        st.session_state.chat_id = new_id
        st.rerun()

if st.session_state.chat_id is None:
    st.write("📭 Create or select a chat from the sidebar.")
    st.stop()

# ─── Main chat UI ────────────────────────────────────────────────────────────
cid = st.session_state.chat_id
chat = load_chat(cid)

st.header(chat.get("title") or "Untitled Chat")

# File selector
all_files = {
    p.stem.split("_",1)[0]: p
    for p in FILE_DIR.iterdir() if p.is_file()
}
attached = st.multiselect(
    "Select PDFs to include",
    options=chat["file_ids"],
    default=chat["file_ids"]
)
chat["file_ids"] = attached

# Show history
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# New user question
q = st.chat_input("Ask your question…")
if q:
    # record user
    chat["messages"].append({"role":"user","content":q})
    if chat["title"] is None:
        chat["title"] = q[:40]

    # prepare pixel_values & prompt
    if attached:
        batches = [pdf_to_tensor_bf16(str(all_files[fid])) for fid in attached]
        pixel_values = torch.cat(batches, dim=0)
        # one leading newline per page
        prompt = "\n" * pixel_values.shape[0] + q
    else:
        pixel_values = None
        prompt = q

    # run chat
    with st.spinner("Generating…"):
        resp = model.chat(
            tok,
            pixel_values,
            prompt,
            generation_config={"max_new_tokens":1024}
        )
    ans = resp["assistant"] if isinstance(resp, dict) else resp

    # record & display
    chat["messages"].append({"role":"assistant","content":ans})
    save_chat(cid, chat)

    with st.chat_message("assistant"):
        st.write(ans)
