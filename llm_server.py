# llm_server.py  ── launch with:  uvicorn llm_server:app --host 0.0.0.0 --port 8000 --workers 1
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError("🚨  GPU not visible - check nvidia-smi")

app       = FastAPI(title="Skywork-R1 V2 Chat API")
tokenizer = None
model     = None

# ---------- load once, pin to GPU ----------
@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        # device_map="auto",           # first visible GPU 
        device_map="balanced_low_0",  # 4 × L40S shards   
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()

    # safe text-only warm-up  (no img token assert)
    _ = model.chat(tokenizer, pixel_values=None,
                   question="ping", generation_config={"max_new_tokens":1})
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"✅  Skywork loaded - {mem:0.1f} GB on {torch.cuda.get_device_name(0)}")

# ---------- schema ----------
class ChatRequest(BaseModel):
    system: str
    user:   str
    max_tokens: int | None = 4096
class ChatResponse(BaseModel):
    answer: str

# ---------- endpoint ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(503, "Model not ready")
    merged = req.system + req.user
    try:
        out = model.chat(tokenizer, pixel_values=None,
                         question=merged,
                         generation_config={"max_new_tokens": req.max_tokens})
        return ChatResponse(answer=out)
    except Exception as err:
        raise HTTPException(500, str(err))
