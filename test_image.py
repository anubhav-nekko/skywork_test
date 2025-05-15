from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPImageProcessor            # ← import the correct image processor
)

MODEL_DIR = "/home/ubuntu/models/Skywork-R1V2-38B"

# 1️⃣ text objects
tok   = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map="balanced_low_0",     # 4 × 48 GB L40S
            torch_dtype="auto",
            trust_remote_code=True           # loads SkyworkChatModel
        ).eval()

# 2️⃣ vision object
image_proc = CLIPImageProcessor.from_pretrained(MODEL_DIR)  # the repo has a CLIP config

# 3️⃣ prepare one image
img = Image.open("cat.jpg").convert("RGB")
pixel_values = image_proc(images=img, return_tensors="pt").pixel_values.to("cuda:0")

# 4️⃣ ask the multimodal model
answer = model.chat(
    tok,
    pixel_values=pixel_values,
    question="What animal is this?",
    generation_config={"max_new_tokens": 1024}
)
print(answer)
