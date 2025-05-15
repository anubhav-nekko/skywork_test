import fitz, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"          # adjust if different
tok  = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="balanced_low_0",   # splits 151 GB FP16 across 4 × 48 GB GPUs :contentReference[oaicite:2]{index=2}
            torch_dtype="auto",
            trust_remote_code=True
        ).eval()

# ── load first N pages as images ─────────────────────────
pages = []
doc = fitz.open("scan.pdf")                                       # PyMuPDF page loader :contentReference[oaicite:3]{index=3}
for p in doc:                                                 # limit to first 5 pages
    pix  = p.get_pixmap(matrix=fitz.Matrix(1, 1))                 # raster page → Pixmap :contentReference[oaicite:4]{index=4}
    pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))

# ── prepare only the image tensors (skip text) ────────────
pixel_values = proc.image_processor(                              # <— key change
                  images=pages, return_tensors="pt"
              ).pixel_values.to("cuda:0")

answer = model.chat(
            tok,
            pixel_values=pixel_values,                            # images only
            question="Summarize the main allegations against Sarda Ji.",
            generation_config={"max_new_tokens":2048}
         )
print(answer)
