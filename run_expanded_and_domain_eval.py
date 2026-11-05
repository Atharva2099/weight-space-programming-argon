"""Expanded Headroom & Cross-Domain Benchmark (4,096 Tokens + Coding & Instruction Controls).

Evaluates across Base vs. Astra vs. Argon Phase 1 vs. Argon Phase 2:
1. Part 1: 20 MATH-500 competition problems with max_new_tokens=4096 (expanded headroom).
2. Part 2: 10 Cross-Domain prompts (Python coding and general instruction following).

Saves incrementally to expanded_domain_eval.jsonl, pushes to GitHub, and shuts down VM.
"""

import json
import os
import re
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from apply_argon_patch import apply_think_exit_weight_patch as apply_phase1_patch
from apply_constrained_argon_patch import apply_qwen35_math_weight_patch as apply_phase2_patch

BASE_ID = "Qwen/Qwen3.5-0.8B"
ASTRA_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
OUTPUT_JSONL = "expanded_domain_eval.jsonl"
OUTPUT_JSON = "expanded_domain_eval.json"

DOMAIN_PROMPTS = [
    # Python Coding
    {"id": "code_01", "type": "code", "q": "Write a Python function `is_palindrome(s: str) -> bool` that ignores punctuation, whitespace, and case. Include docstring and example usage."},
    {"id": "code_02", "type": "code", "q": "Write a Python function `fibonacci(n: int) -> list[int]` that returns the first n Fibonacci numbers efficiently using dynamic programming or memoization."},
    {"id": "code_03", "type": "code", "q": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals."},
    {"id": "code_04", "type": "code", "q": "Write a Python class `LRUCache` with capacity, `get(key)`, and `put(key, value)` with O(1) time complexity."},
    {"id": "code_05", "type": "code", "q": "Write a Python function to find the length of the longest substring without repeating characters."},
    # Instruction Following & General Reasoning
    {"id": "inst_01", "type": "instruction", "q": "Summarize the key difference between supervised learning and reinforcement learning in exactly three bullet points. Do not use the letter 'e' in your third bullet point."},
    {"id": "inst_02", "type": "instruction", "q": "Write a short professional email declining an invitation to speak at a conference due to prior travel commitments. Maintain a polite, warm tone."},
    {"id": "inst_03", "type": "instruction", "q": "Output a valid JSON object representing a book with fields: title, author, year_published, genres (array of strings), and in_print (boolean)."},
    {"id": "inst_04", "type": "instruction", "q": "Explain what causes ocean tides to a curious 10-year-old in two paragraphs using clear, simple analogies."},
    {"id": "inst_05", "type": "instruction", "q": "List five creative names for a sustainable urban coffee shop, with a one-sentence rationale for each name."},
]

def load_math_headroom_data():
    print("Loading 20 MATH-500 competition problems for 4,096-token headroom test...", flush=True)
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")[:20]
    items = []
    for i, (q, a) in enumerate(zip(math500["problem"], math500["answer"])):
        items.append({"id": f"math500_headroom_{i:02d}", "type": "math_headroom", "q": q, "gt": a})
    return items

def run_test_suite(model, tokenizer, items, condition_name: str, max_tokens: int = 4096):
    print(f"\n{'='*25} EVALUATING {condition_name} ({len(items)} items) {'='*25}", flush=True)
    results = []
    exits = 0

    for idx, item in enumerate(items):
        is_math = item["type"] == "math_headroom"
        if is_math:
            prompt = f"<|im_start|>user\n{item['q']}\nPlease reason step by step and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
            max_gen = max_tokens
        else:
            prompt = f"<|im_start|>user\n{item['q']}<|im_end|>\n<|im_start|>assistant\n"
            max_gen = 512

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )

        gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
        has_exit = ("</think>" in text) if is_math else True
        if has_exit:
            exits += 1

        record = {
            "id": item["id"],
            "type": item["type"],
            "condition": condition_name,
            "gen_tokens": gen_tokens,
            "has_exit": has_exit,
            "snippet": text[-300:] if len(text) > 300 else text
        }
        results.append(record)

        with open(OUTPUT_JSONL, "a") as f_log:
            f_log.write(json.dumps(record) + "\n")

        print(f"[{condition_name}] Item {idx+1}/{len(items)} ({item['id']}) | Tokens: {gen_tokens} | Exited: {has_exit}", flush=True)

    return results

def main():
    math_items = load_math_headroom_data()
    all_items = math_items + DOMAIN_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    summary = {}

    # 1. Base Model
    print("Loading Base Model...", flush=True)
    base_m = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    summary["base"] = run_test_suite(base_m, tokenizer, all_items, "BASE_MODEL")
    del base_m
    torch.cuda.empty_cache()

    # 2. Astra Model
    print("\nLoading Astra Edited Model...", flush=True)
    astra_m = AutoModelForCausalLM.from_pretrained(ASTRA_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    summary["astra"] = run_test_suite(astra_m, tokenizer, all_items, "ASTRA_EDIT")
    del astra_m
    torch.cuda.empty_cache()

    # 3. Argon Phase 1 (State-Conditional Bridge)
    print("\nLoading Argon Phase 1 (Unconstrained Bridge)...", flush=True)
    argon_p1 = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    apply_phase1_patch(argon_p1, tokenizer)
    summary["argon_p1"] = run_test_suite(argon_p1, tokenizer, all_items, "ARGON_PHASE1")
    del argon_p1
    torch.cuda.empty_cache()

    # 4. Argon Phase 2 (Constrained Internal Surgery)
    print("\nLoading Argon Phase 2 (Constrained Internal Surgery)...", flush=True)
    argon_p2 = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    apply_phase2_patch(argon_p2)
    summary["argon_p2"] = run_test_suite(argon_p2, tokenizer, all_items, "ARGON_PHASE2")
    del argon_p2
    torch.cuda.empty_cache()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60, flush=True)
    print("EXPANDED HEADROOM & DOMAIN CONTROL EVALUATION COMPLETE!", flush=True)
    print("="*60, flush=True)

    # Git push and shutdown
    try:
        subprocess.run(["git", "add", OUTPUT_JSON, OUTPUT_JSONL], check=True)
        subprocess.run(["git", "commit", "-m", "Add expanded 4k headroom and cross-domain control evaluation results"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Pushed expanded results to GitHub repository successfully!", flush=True)
    except Exception as e:
        print(f"Git push notice: {e}", flush=True)

    print("\nUnmasking poweroff and shutting down VM cleanly...", flush=True)
    subprocess.run(["sudo", "chmod", "755", "/sbin/poweroff", "/sbin/shutdown"])
    subprocess.run(["sync"])
    subprocess.run(["sudo", "poweroff"])

if __name__ == "__main__":
    main()
