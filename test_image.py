from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir = "/home/ubuntu/models/Skywork-R1V2-38B"

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="balanced_low_0",        # spreads 151 GB FP16 over 4 × 48 GB GPUs :contentReference[oaicite:10]{index=10}
        torch_dtype="auto",
        trust_remote_code=True              # ← loads the custom skywork_chat class
).eval()

proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

img = Image.open("cat.jpg")
answer = model.chat(tok,
                    pixel_values=proc(images=[img], return_tensors="pt").pixel_values.to("cuda:0"),
                    question="What animal is this?",
                    generation_config={"max_new_tokens":64})
print(answer)