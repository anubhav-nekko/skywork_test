# llm_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, uvicorn, os

###############################################################################
# ―― 1.  Load once at start-up and pin to GPU ――
###############################################################################
MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"   # sanity check
if DEVICE == "cpu":
    raise RuntimeError("GPU not visible – check drivers & nvidia-smi")

tokenizer = None
model     = None

app = FastAPI(
    title="Skywork-R1 V2 Chat API",
    description="GPU-resident inference service",
    version="0.1.0"
)

@app.on_event("startup")
def load_model():
    """Runs **once** when the Uvicorn worker boots."""
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",          # First GPU only → balanced_low_0 not needed
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()
    # Optional - warm-up
    _ = model.generate(**tokenizer("hello", return_tensors="pt").to(model.device),
                       max_new_tokens=1)
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"Skywork loaded → {gpu_mem:0.1f} GB on {torch.cuda.get_device_name(0)}")

###############################################################################
# ―― 2.  API schema & endpoint ――
###############################################################################
class ChatRequest(BaseModel):
    system: str
    user:   str
    max_tokens: int | None = 4096

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    prompt = req.system + req.user
    try:
        answer = model.chat(
            tokenizer, pixel_values=None,
            question=prompt,
            generation_config={"max_new_tokens": req.max_tokens}
        )
        return ChatResponse(answer=answer)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
###############################################################################
# ―― 3.  Run with **one** worker so the model is loaded once ――
###############################################################################
if __name__ == "__main__":
    uvicorn.run("llm_server:app", host="0.0.0.0", port=8000, workers=1)
