"""Phase 2 Full 100-Problem Benchmark: Constrained Internal-Only Patch.

Evaluates Argon's Phase 2 constrained patch across 100 official benchmark problems:
- 50 from HuggingFaceH4/MATH-500
- 50 from openai/gsm8k

Features incremental per-problem line flushing to JSONL, automated Git push
to GitHub, and automatic VM poweroff upon completion.
"""

import os
import json
import re
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from apply_constrained_argon_patch import apply_qwen35_math_weight_patch

MODEL_ID = "Qwen/Qwen3.5-0.8B"
OUTPUT_JSON = "phase2_benchmark_results.json"
OUTPUT_JSONL = "phase2_benchmark_results.jsonl"

def extract_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else ""

def load_data():
    print("Loading official benchmark datasets (50 MATH-500, 50 GSM8K)...", flush=True)
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")[:50]
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")[:50]
    
    problems = []
    for i, (q, a) in enumerate(zip(math500["problem"], math500["answer"])):
        problems.append({"id": f"math500_{i:02d}", "dataset": "MATH-500", "q": q, "gt": a})
    for i, (q, a) in enumerate(zip(gsm8k["question"], gsm8k["answer"])):
        gt_num = a.split("####")[-1].strip() if "####" in a else a.strip()
        problems.append({"id": f"gsm8k_{i:02d}", "dataset": "GSM8K", "q": q, "gt": gt_num})
    
    print(f"Loaded {len(problems)} problems successfully.", flush=True)
    return problems

def main():
    problems = load_data()
    print(f"Loading {MODEL_ID} on CUDA...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    print("\nApplying Phase 2 Constrained Internal-Only Weight Patch...", flush=True)
    apply_qwen35_math_weight_patch(model)
    print("Patch applied successfully (Layers 16-23 down_proj + Layer 23 o_proj).", flush=True)

    print("\n" + "="*30 + " EVALUATING PHASE 2 CONSTRAINED PATCH " + "="*30, flush=True)
    results = []
    exits = 0

    for idx, p in enumerate(problems):
        formatted = f"<|im_start|>user\n{p['q']}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
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
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
        has_exit = "</think>" in text
        if has_exit:
            exits += 1
            parts = text.split("</think>", 1)
            final_resp = parts[1].strip()
        else:
            final_resp = "[TRUNCATED - NEVER EXITED THINKING]"

        ans = extract_answer(final_resp)
        record = {
            "id": p["id"],
            "dataset": p["dataset"],
            "condition": "PHASE2_CONSTRAINED",
            "exited": has_exit,
            "pred": ans,
            "gt": p["gt"],
            "gen_tokens": gen_tokens,
            "resp_sample": final_resp[:180]
        }
        results.append(record)

        # Atomic incremental flush per problem
        with open(OUTPUT_JSONL, "a") as f_log:
            f_log.write(json.dumps(record) + "\n")

        if (idx + 1) % 5 == 0 or (idx + 1) == len(problems):
            print(f"[PHASE2] Progress: {idx+1}/{len(problems)} | Current Exit Rate: {exits/(idx+1)*100:.1f}%", flush=True)

    exit_rate = exits / len(problems)
    summary = {
        "condition": "PHASE2_CONSTRAINED",
        "total_problems": len(problems),
        "exits": exits,
        "exit_rate": exit_rate,
        "records": results
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60, flush=True)
    print(f"PHASE 2 CONSTRAINED BENCHMARK COMPLETE: {exits}/{len(problems)} Exited ({exit_rate*100:.1f}%)", flush=True)
    print("="*60, flush=True)

    # Automated GitHub push from VM
    print("\nCommitting and pushing results to GitHub...", flush=True)
    try:
        subprocess.run(["git", "add", OUTPUT_JSON, OUTPUT_JSONL], check=True)
        subprocess.run(["git", "commit", "-m", f"Add Phase 2 100-problem benchmark results ({exit_rate*100:.1f}% exit rate)"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("Successfully pushed benchmark results to GitHub repository!", flush=True)
    except Exception as e:
        print(f"Git push notice: {e}", flush=True)

    # Power off VM safely
    print("\nSyncing storage and powering off VM...", flush=True)
    subprocess.run(["sync"])
    subprocess.run(["sudo", "poweroff"])

if __name__ == "__main__":
    main()
