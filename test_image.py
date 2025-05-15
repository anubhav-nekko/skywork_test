from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"

tok   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
proc  = AutoProcessor.from_pretrained(MODEL_DIR,  trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="balanced_low_0",     # 4 × L40S shards
            torch_dtype="auto",
            trust_remote_code=True           # loads skywork_chat class
        ).eval()

img     = Image.open("cat.jpg").convert("RGB")   # any RGB image
question = "What animal is this?"

# 1️⃣  Process **only** the image
pixel_values = proc(images=[img], return_tensors="pt").pixel_values.to("cuda:0")

# 2️⃣  Ask the model
answer = model.chat(
    tok,
    pixel_values=pixel_values,
    question=question,
    generation_config={"max_new_tokens": 64}
)
print(answer)
