"""Three-Way Head-to-Head Benchmark: Base vs. Astra vs. Argon on MATH-500 & GSM8K."""

import json
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from apply_argon_patch import apply_think_exit_weight_patch

BASE_ID = "Qwen/Qwen3.5-0.8B"
ASTRA_ID = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"

def extract_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return nums[-1] if nums else ""

def load_data():
    print("Loading official benchmark datasets...")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")[:50]
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")[:50]
    
    problems = []
    for i, (q, a) in enumerate(zip(math500["problem"], math500["answer"])):
        problems.append({"id": f"math500_{i:02d}", "dataset": "MATH-500", "q": q, "gt": a})
    for i, (q, a) in enumerate(zip(gsm8k["question"], gsm8k["answer"])):
        gt_num = a.split("####")[-1].strip() if "####" in a else a.strip()
        problems.append({"id": f"gsm8k_{i:02d}", "dataset": "GSM8K", "q": q, "gt": gt_num})
    
    print(f"Loaded {len(problems)} total problems (50 MATH-500, 50 GSM8K).")
    return problems

def evaluate_model(model, tokenizer, problems, condition_name: str):
    print(f"\n{'='*25} EVALUATING {condition_name} {'='*25}")
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

        text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
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
            "condition": condition_name,
            "exited": has_exit,
            "pred": ans,
            "gt": p["gt"],
            "resp_sample": final_resp[:180]
        }
        results.append(record)
        
        # Incremental checkpointing: append immediately to jsonl so no progress is ever lost
        with open("benchmark_results_3way.jsonl", "a") as f_log:
            f_log.write(json.dumps(record) + "\n")

        if (idx + 1) % 10 == 0:
            print(f"[{condition_name}] Progress: {idx+1}/{len(problems)} (Current Exit Rate: {exits/(idx+1)*100:.1f}%)", flush=True)

    exit_rate = exits / len(problems)
    print(f"[{condition_name}] FINAL: {exits}/{len(problems)} exited ({exit_rate*100:.1f}%)")
    return {"condition": condition_name, "exit_rate": exit_rate, "records": results}

if __name__ == "__main__":
    problems = load_data()
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    summary = {}

    # 1. Base Model
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    summary["base"] = evaluate_model(base_model, tokenizer, problems, "BASE_MODEL")
    del base_model
    torch.cuda.empty_cache()

    # 2. Astra Model
    print("\nLoading Astra Edited Model...")
    astra_model = AutoModelForCausalLM.from_pretrained(ASTRA_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    summary["astra"] = evaluate_model(astra_model, tokenizer, problems, "ASTRA_EDIT")
    del astra_model
    torch.cuda.empty_cache()

    # 3. Argon Model
    print("\nLoading Base Model for Argon Patch...")
    argon_model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    apply_think_exit_weight_patch(argon_model, tokenizer)
    summary["argon"] = evaluate_model(argon_model, tokenizer, problems, "ARGON_EDIT")

    with open("benchmark_results_3way.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("THREE-WAY BENCHMARK HEAD-TO-HEAD SUMMARY (100 Problems):")
    print(f"  1. Base Model Exit Rate:         {summary['base']['exit_rate']*100:.1f}%")
    print(f"  2. GPT-6 Astra Edit Exit Rate:   {summary['astra']['exit_rate']*100:.1f}%")
    print(f"  3. Gemini 4 Argon Edit Exit Rate: {summary['argon']['exit_rate']*100:.1f}%")
    print("=" * 60)
