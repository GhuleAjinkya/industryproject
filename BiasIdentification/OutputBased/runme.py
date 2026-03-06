import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import accelerate
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

try:
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
    print("generated text:", tokenizer.decode(generated[0], skip_special_tokens=True))
except Exception as e:
    print("generation attempt failed:", e)