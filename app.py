
import os, json, time, uuid, io, fitz, torch, torchvision.transforms as T
from pathlib import Path
from PIL import Image
import streamlit as st
from transformers import AutoTokenizer, AutoModel

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STO_DIR    = BASE_DIR / "storage"
FILE_DIR   = STO_DIR / "files"
CHAT_DIR   = STO_DIR / "chats"
META_PATH  = STO_DIR / "meta.json"
MODEL_DIR  = "/home/ubuntu/models/Skywork-R1V2-38B"

# ─── Ensure folders ─────────────────────────────────────────────────────────
for p in (FILE_DIR, CHAT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ─── Simple metadata helpers ────────────────────────────────────────────────
def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"chat_ids": []}

def save_meta(meta):
    META_PATH.write_text(json.dumps(meta))

meta = load_meta()

# ─── Model (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True, hash_funcs={torch.device: str})
def load_skywork():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        device_map="balanced_low_0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()
    return tok, model

tok, model = load_skywork()

# ─── Vision transform (same as reference) ───────────────────────────────────
VISION_TX = T.Compose([
    T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─── PDF → tensor helper ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def pdf_to_tensor(pdf_path: Path):
    doc = fitz.open(pdf_path)
    tensors = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
        tensors.append(VISION_TX(pil))
    return torch.stack(tensors).to(torch.float16)

# ─── Chat persistence helpers ───────────────────────────────────────────────
def new_chat_id() -> str:
    return uuid.uuid4().hex[:8]

def load_chat(chat_id: str):
    p = CHAT_DIR / f"{chat_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"title": None, "messages": [], "file_ids": []}

def save_chat(chat_id: str, data):
    (CHAT_DIR / f"{chat_id}.json").write_text(json.dumps(data))

# ─── Streamlit Session Setup ────────────────────────────────────────────────
if "chat_id" not in st.session_state:
    if meta["chat_ids"]:
        st.session_state.chat_id = meta["chat_ids"][0]   # most recent
    else:
        st.session_state.chat_id = None

# ─── Sidebar UI ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chat")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        # save files and register IDs
        for f in uploaded_files:
            fid = uuid.uuid4().hex[:8]
            dest = FILE_DIR / f"{fid}_{f.name}"
            dest.write_bytes(f.read())
            if st.session_state.chat_id:
                chat_data = load_chat(st.session_state.chat_id)
                chat_data["file_ids"].append(fid)
                save_chat(st.session_state.chat_id, chat_data)

    # existing chats buttons
    st.markdown("### Previous chats")
    for cid in meta["chat_ids"]:
        label = load_chat(cid)["title"] or cid
        if st.button(label, key=f"chat_{cid}"):
            st.session_state.chat_id = cid

    if st.button("➕ New chat"):
        new_id = new_chat_id()
        meta["chat_ids"].insert(0, new_id)
        save_meta(meta)
        save_chat(new_id, {"title": None, "messages": [], "file_ids": []})
        st.session_state.chat_id = new_id

# no active chat → stop
if st.session_state.chat_id is None:
    st.stop()

chat_data = load_chat(st.session_state.chat_id)

# ─── Main Chat UI ───────────────────────────────────────────────────────────
st.header(chat_data.get("title") or "Untitled chat")

# multiselect files attached to chat
all_files = {p.stem.split("_",1)[0]: p for p in FILE_DIR.iterdir() if p.is_file()}
file_options = [fid for fid in chat_data["file_ids"] if fid in all_files]
selected_fids = st.multiselect("Select PDFs to include", file_options, default=file_options)

# render previous messages
for m in chat_data["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Ask your question…")
if user_q:
    # display user msg immediately
    with st.chat_message("user"):
        st.write(user_q)
    chat_data["messages"].append({"role": "user", "content": user_q})
    if chat_data["title"] is None:
        chat_data["title"] = user_q[:40]

    # build vision tensor batch from selected PDFs
    if selected_fids:
        batches = [pdf_to_tensor(all_files[fid]) for fid in selected_fids]
        pixel_values = torch.cat(batches, dim=0).cuda()
        prompt = "\n" * pixel_values.shape[0] + user_q
    else:
        pixel_values = None
        prompt = user_q

    # roll previous assistant / user turns into history text
    history_text = "\n".join(m["content"] for m in chat_data["messages"] if m["role"] == "assistant")

    full_prompt = (prompt if pixel_values is None else prompt)  # prompt already has newlines for images

    with st.spinner("Generating…"):
        answer = model.chat(tok, pixel_values, full_prompt, generation_config=dict(max_new_tokens=256))

    chat_data["messages"].append({"role": "assistant", "content": answer})
    save_chat(st.session_state.chat_id, chat_data)

    with st.chat_message("assistant"):
        st.write(answer)
