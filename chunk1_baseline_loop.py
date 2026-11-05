"""Chunk 1: Baseline run on Qwen3.5-0.8B to observe thinking loop behavior."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"

print(f"Loading {MODEL_ID} on CUDA...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# A standard competition math prompt from MATH
prompt = (
    "A math competition problem:\n"
    "Find all real solutions x to the equation: sqrt(x + 3) + sqrt(x - 2) = 5.\n"
    "Show your step-by-step reasoning.\n"
)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda")

print("Generating response with thinking enabled (sampling: temp=0.6, top_p=0.95, top_k=20)...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

input_len = inputs.input_ids.shape[1]
generated_ids = output_ids[0, input_len:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n" + "=" * 50)
print("GENERATED REASONING (first 1024 tokens):")
print("=" * 50)
print(response)
print("=" * 50)
