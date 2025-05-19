# llm_server.py ── launch with: uvicorn llm_server:app --host 0.0.0.0 --port 8000 --workers 1
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
        device_map="balanced",  # 4 × L40S shards
        torch_dtype="auto",
        trust_remote_code=True
    ).eval()

    # Ensure tokenizer has a chat_template, otherwise, the below formatting might fail
    if tokenizer.chat_template is None:
        # Fallback or specific Skywork template if known and not auto-configured
        # This part might need adjustment based on Skywork's exact setup if
        # tokenizer.chat_template isn't automatically set from their files.
        # For now, we'll proceed assuming it might be set, or that model.chat
        # can handle a simple concatenation if the template is truly missing.
        # However, the goal is to use the model's intended format.
        print("⚠️ Warning: Tokenizer does not have a chat_template. Prompt formatting might be suboptimal.")


    # safe text-only warm-up (no img token assert)
    # For warm-up, a simple ping is fine, or a templated one if required.
    warmup_messages = [{"role": "user", "content": "ping"}]
    try:
        # Attempt to use chat template for warmup as well for consistency
        formatted_warmup_prompt = tokenizer.apply_chat_template(
            warmup_messages,
            tokenize=False,
            add_generation_prompt=True # Usually True for generation
        )
    except Exception as e:
        # If templating fails for warmup, fall back to simpler ping
        print(f"Warning: Could not apply chat template for warm-up: {e}. Using simple ping.")
        formatted_warmup_prompt = "ping"

    _ = model.chat(tokenizer, pixel_values=None,
                   question=formatted_warmup_prompt,
                   generation_config={"max_new_tokens": 1})
    mem = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem += torch.cuda.memory_allocated(i)
    mem /= 1e9
    print(f"✅  Skywork loaded - {mem:0.1f} GB on {torch.cuda.device_count()} GPU(s)")

# ---------- schema ----------
class ChatRequest(BaseModel):
    system: str | None = None # Make system prompt optional
    user:   str
    max_tokens: int | None = 4096

class ChatResponse(BaseModel):
    answer: str

# ---------- endpoint ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None or tokenizer is None: # also check tokenizer
        raise HTTPException(503, "Model or tokenizer not ready")

    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.user})

    try:
        # Apply the chat template
        # `add_generation_prompt=True` is typically used to ensure the prompt ends
        # in a way that signals to the model it should start generating.
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback if chat templating fails (though it ideally shouldn't with trust_remote_code=True)
        print(f"Error applying chat template: {e}. Falling back to simple concatenation.")
        formatted_prompt = (req.system + " " + req.user) if req.system else req.user


    try:
        out = model.chat(tokenizer, pixel_values=None,
                         question=formatted_prompt,
                         generation_config={"max_new_tokens": req.max_tokens})
        return ChatResponse(answer=out)
    except Exception as err:
        print(f"Error during model.chat: {err}") # Log the error server-side
        raise HTTPException(500, str(err))