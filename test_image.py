from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"

tok     = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
proc    = AutoProcessor.from_pretrained(MODEL_DIR,  trust_remote_code=True)
model   = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="balanced_low_0",     # 4 × L40S
            torch_dtype="auto",
            trust_remote_code=True
          ).eval()

img     = Image.open("cat.jpg").convert("RGB")
prompt  = "What animal is this?"

# 1️⃣  Build **joint** inputs – give BOTH image *and* text
inputs = proc(images=[img], text=[prompt], return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}        # move to GPU

# 2️⃣  Generate
output_ids = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(output_ids[0], skip_special_tokens=True))
