"""Benchmark Evaluation: Base Qwen3.5-0.8B vs. Argon Patched Model."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apply_argon_patch import apply_think_exit_weight_patch

MODEL_ID = "Qwen/Qwen3.5-0.8B"

BENCHMARK_PROBLEMS = [
    # Algebra
    {"id": "math_01", "q": "Find all real solutions x to: sqrt(2x + 5) - sqrt(x - 1) = 2.", "a": "2 or 10"},
    {"id": "math_02", "q": "If 3^(2x - 1) = 27^(x + 2), find the value of x.", "a": "-7"},
    {"id": "math_03", "q": "Find the sum of all roots of the equation x^3 - 6x^2 + 11x - 6 = 0.", "a": "6"},
    # Number Theory
    {"id": "math_04", "q": "What is the remainder when 7^100 is divided by 5?", "a": "1"},
    {"id": "math_05", "q": "Find the greatest common divisor of 252 and 198.", "a": "18"},
    # Word Problems & Combinatorics
    {"id": "math_06", "q": "A store sells shirts for $20 each and jackets for $50 each. If a customer buys 8 items and spends $280, how many jackets did they buy?", "a": "4"},
    {"id": "math_07", "q": "How many positive three-digit integers are multiples of both 3 and 4?", "a": "75"},
    {"id": "math_08", "q": "In a group of 30 people, 18 play soccer and 15 play basketball. If 5 play neither, how many play both?", "a": "8"},
    # Geometry & Functions
    {"id": "math_09", "q": "A right triangle has legs of length 9 and 12. Find the length of the altitude drawn to the hypotenuse.", "a": "7.2 (or 36/5)"},
    {"id": "math_10", "q": "If f(x) = x^2 - 4x + 7, find the minimum value of f(x) for all real x.", "a": "3"},
]

def run_evaluation(model, tokenizer, is_patched: bool = False):
    label = "ARGON_PATCHED" if is_patched else "BASE_MODEL"
    print(f"\n{'='*20} RUNNING {label} {'='*20}")
    
    exited_count = 0
    token_capped_count = 0
    results = []

    for idx, item in enumerate(BENCHMARK_PROBLEMS):
        prompt = item["q"]
        formatted = f"<|im_start|>user\n{prompt}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        inputs = tokenizer([formatted], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )

        gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        has_exit = "</think>" in text
        
        if has_exit:
            exited_count += 1
            parts = text.split("</think>", 1)
            think_len = len(parts[0])
            final_resp = parts[1].strip()
        else:
            token_capped_count += 1
            think_len = len(text)
            final_resp = "[TRUNCATED - NEVER EXITED THINKING]"

        print(f"[{item['id']}] Exited: {has_exit} | Gen Tokens: {gen_tokens} | Think chars: {think_len}")
        results.append({
            "id": item["id"],
            "exited": has_exit,
            "gen_tokens": gen_tokens,
            "final_resp": final_resp[:300]
        })

    print(f"\n{label} SUMMARY: {exited_count}/{len(BENCHMARK_PROBLEMS)} exited thinking ({exited_count/len(BENCHMARK_PROBLEMS)*100:.1f}%)")
    return {"label": label, "exit_rate": exited_count / len(BENCHMARK_PROBLEMS), "results": results}

if __name__ == "__main__":
    print(f"Loading {MODEL_ID} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    # 1. Base Model Run
    base_res = run_evaluation(model, tokenizer, is_patched=False)

    # 2. Apply Argon's Patch
    print("\nApplying Argon's Weight-Space Programming Patch...")
    patch_info = apply_think_exit_weight_patch(model, tokenizer)
    print("Patch applied:", patch_info)

    # 3. Patched Model Run
    patched_res = run_evaluation(model, tokenizer, is_patched=True)

    print("\n" + "=" * 60)
    print("FINAL BENCHMARK COMPARISON:")
    print(f"  Base Model Exit Rate:         {base_res['exit_rate']*100:.1f}%")
    print(f"  Argon Patched Model Exit Rate: {patched_res['exit_rate']*100:.1f}%")
    print("=" * 60)
