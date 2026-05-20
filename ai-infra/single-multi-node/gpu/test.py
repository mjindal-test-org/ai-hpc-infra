from huggingface_hub import model_info
info = model_info("mistralai/Mistral-7B-v0.3")
print("Gated:", info.gated)
print("You have access:", info.gated == False or "approved" in str(info))