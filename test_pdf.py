# test_pdf.py  –  Skywork‑R1 V2 38B, multi‑page PDF inference
import torch, torchvision.transforms as T, fitz
from PIL import Image
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"   # adjust if different

# ── 1.  load tokenizer + sharded model (4×L40S) ───────────────────────────
tok   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)   # :contentReference[oaicite:0]{index=0}
model = AutoModel.from_pretrained(
           MODEL_DIR,
           device_map="balanced_low_0",      # spreads 151 GB over 4 GPUs :contentReference[oaicite:1]{index=1}
           torch_dtype=torch.float16,
           trust_remote_code=True            # loads skywork_chat class
        ).eval()

# ── 2.  open a PDF and rasterise the first N pages (PyMuPDF) ──────────────
pdf_path   = "scan.pdf"
# max_pages  = 5                               # change as needed
doc        = fitz.open(pdf_path)             # :contentReference[oaicite:2]{index=2}
pages      = [page for page in doc]

# ── 3.  page‑to‑PIL and vision transform (448×448 + normalise) ────────────
tx = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),                       # HWC ➜ CHW, [0,1]
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ])

img_tensors = []
for p in pages:
    pix   = p.get_pixmap(matrix=fitz.Matrix(1, 1))   # raster @72 dpi  :contentReference[oaicite:3]{index=3}
    pil   = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")
    img_tensors.append(tx(pil))

pixel_values = torch.stack(img_tensors).to(torch.float16).cuda()   # [B,C,H,W]

# ── 4.  build prompt (one leading newline per image, per InternVL spec) ──
prompt = "\n" * len(img_tensors) + "Summarise the key points in this document."

# ── 5.  ask the model – *no processor call needed* ───────────────────────
answer = model.chat(
            tok,
            pixel_values,
            prompt,
            generation_config=dict(max_new_tokens=256)
         )
print("Assistant:", answer)
