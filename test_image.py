# test_ok.py  –  Skywork-R1 V2 single-image inference on g6e.12xlarge
import torch, torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, AutoModel

model_dir = "/home/ubuntu/models/Skywork-R1V2-38B"

# 1️⃣ load model + tokenizer
tok   = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(
           model_dir,
           device_map="balanced_low_0",     # 4 × L40S shards
           torch_dtype=torch.float16,
           trust_remote_code=True
        ).eval()

# 2️⃣ load & preprocess a 448×448 RGB image
img = Image.open("cat.jpg").convert("RGB").resize((448, 448))
pixel_values = T.Compose([
    T.ToTensor(),                           # HWC→CHW, [0,1]
    T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])(img).unsqueeze(0).to(torch.float16).cuda()

# 3️⃣ build the prompt (one leading \n because there is one image)
prompt = "\nWhat animal is this?"

# 4️⃣ call model.chat(tokenizer, pixel_values, prompt, ...)
answer = model.chat(
            tok,
            pixel_values,
            prompt,
            generation_config=dict(max_new_tokens=1024)
         )
print("Assistant:", answer)
