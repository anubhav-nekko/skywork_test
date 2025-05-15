# app.py  –  Skywork-R1 V2 PDF chat MVP
import os, json, uuid, torch, fitz, torchvision.transforms as T
from pathlib import Path
from PIL import Image
import streamlit as st
from transformers import AutoTokenizer, AutoModel

# ── CONFIG ───────────────────────────────────────────────
MODEL_DIR   = "/home/ubuntu/models/Skywork-R1V2-38B"
BASE_DIR    = Path(__file__).parent.resolve()
UPLOADS_DIR = BASE_DIR / "uploads"
META_FILE   = BASE_DIR / "metadata.json"
CHAT_MAX_TOK = 1024

# ── FOLDERS ──────────────────────────────────────────────
UPLOADS_DIR.mkdir(exist_ok=True)
if not META_FILE.exists():
    META_FILE.write_text(json.dumps({"chats": []}))

# ── MODEL (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    tok   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        device_map="balanced_low_0",
        torch_dtype=torch.float16
    ).eval()
    return tok, model
tok, model = load_model()

# ── VISION TRANSFORM ────────────────────────────────────
VISION_TX = T.Compose([
    T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def pdf_to_tensors(path: Path):
    """Return list[Tensor] for all pages in a PDF."""
    pages = []
    doc = fitz.open(path)
    for pg in doc:
        pix = pg.get_pixmap(matrix=fitz.Matrix(1, 1))
        pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(VISION_TX(pil))
    return pages

# ── METADATA HELPERS ────────────────────────────────────
def load_meta():
    return json.loads(META_FILE.read_text())

def save_meta(meta):
    META_FILE.write_text(json.dumps(meta, indent=2))

meta = load_meta()

# ── SESSION STATE INIT ──────────────────────────────────
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

def new_chat():
    cid = uuid.uuid4().hex[:8]
    meta["chats"].insert(0, {"id": cid, "title": "", "files": [], "history": []})
    save_meta(meta)
    st.session_state.chat_id = cid

# ── SIDEBAR (uploads + past chats) ──────────────────────
with st.sidebar:
    st.title("📄 PDF Chat")
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            dest = UPLOADS_DIR / f.name
            dest.write_bytes(f.read())
            st.success(f"Saved {f.name}")
    st.divider()
    st.markdown("### Conversations")
    for ch in meta["chats"]:
        label = ch["title"] or ch["id"]
        if st.button(label, key=ch["id"]):
            st.session_state.chat_id = ch["id"]
    if st.button("➕ New Chat"):
        new_chat()

# ── NO CHAT SELECTED ────────────────────────────────────
if st.session_state.chat_id is None:
    st.stop()

# ── CURRENT CHAT OBJECT ─────────────────────────────────
chat = next(c for c in meta["chats"] if c["id"] == st.session_state.chat_id)

st.header(chat["title"] or "Untitled chat")

# ── FILE SELECTION ──────────────────────────────────────
pdf_files = sorted(p.name for p in UPLOADS_DIR.iterdir() if p.suffix.lower()==".pdf")
selected = st.multiselect("Include PDFs:", pdf_files, default=chat["files"])
chat["files"] = selected

# ── DISPLAY HISTORY ─────────────────────────────────────
for msg in chat["history"]:
    role = "You" if msg["role"] == "user" else "Assistant"
    st.markdown(f"**{role}:** {msg['text']}")

# ── USER INPUT ──────────────────────────────────────────
user_q = st.text_input("Ask a question:")
if st.button("Send") and user_q:
    chat["history"].append({"role":"user","text":user_q})
    st.markdown(f"**You:** {user_q}")

    # build vision tensor batch
    if selected:
        batches = []
        for fn in selected:
            batches.extend(pdf_to_tensors(UPLOADS_DIR / fn))
        pixel_values = torch.stack(batches).to(torch.float16).cuda()
        prompt = "\n" * pixel_values.shape[0] + user_q
    else:
        pixel_values = None
        prompt = user_q

    with st.spinner("Thinking…"):
        answer = model.chat(tok, pixel_values, prompt,
                            generation_config={"max_new_tokens": CHAT_MAX_TOK})

    chat["history"].append({"role":"assistant","text":answer})
    if not chat["title"]:
        chat["title"] = user_q[:40]
    save_meta(meta)
    st.markdown(f"**Assistant:** {answer}")
