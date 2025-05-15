from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"

tok   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
proc  = AutoProcessor.from_pretrained(MODEL_DIR,  trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
           MODEL_DIR,
           device_map="balanced_low_0",          # 4×L40S shards
           torch_dtype="auto",
           trust_remote_code=True                # loads skywork_chat class
        ).eval()

img      = Image.open("cat.jpg").convert("RGB")
question = "What animal is this?"

# 👉 the **only** correct way to build inputs
inputs = proc(text=question, images=img, return_tensors="pt").to("cuda:0")

out_ids = model.generate(**inputs, max_new_tokens=1024)
print(tok.decode(out_ids[0], skip_special_tokens=True))
