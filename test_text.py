from transformers import AutoModelForCausalLM, AutoTokenizer
model_dir = "/home/ubuntu/models/Skywork-R1V2-38B"

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="balanced_low_0",        # spreads 151 GB FP16 over 4 × 48 GB GPUs :contentReference[oaicite:10]{index=10}
        torch_dtype="auto",
        trust_remote_code=True              # ← loads the custom skywork_chat class
).eval()

prompt = "Explain Newton's first law in two sentences."
# **use the helper**; no pixel_values so text-only:
answer = model.chat(tok, pixel_values=None,
                    question=prompt,
                    generation_config={"max_new_tokens":1024})
print(answer)
