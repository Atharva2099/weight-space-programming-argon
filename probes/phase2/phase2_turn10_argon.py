import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()

# Forward pass counter (strictly <= 14)
forward_passes_used = 0
MAX_BUDGET = 14

def run_generate(prompt_text, max_new_tokens=320):
    global forward_passes_used
    assert forward_passes_used < MAX_BUDGET, "Budget exceeded!"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    forward_passes_used += 1
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False), len(gen_ids)

# 1. Construct the contrastive steering target in logit/embedding space (0 forward passes)
embed_W = model.model.embed_tokens.weight.detach().float()

def get_token_vec(words):
    vecs = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        for tid in ids:
            vecs.append(embed_W[tid])
    return torch.stack(vecs, dim=0).mean(dim=0)

concise_close_words = [
    "</think>", "**Final", "Final Answer", "Therefore", "Thus,", "Answer:", "\n</think>\n"
]
loop_filler_words = [
    "Wait,", "Wait", "Let me check", "Let's double check", "Hmm,", "Alternatively,", "Let me recalculate"
]

v_close = get_token_vec(concise_close_words)
v_loop = get_token_vec(loop_filler_words)
v_steer = v_close - v_loop
v_steer = v_steer / (v_steer.norm() + 1e-8)

# 2. Compute targeted rank-1 + top-singular-channel updates on mid-to-late MLP down_proj
target_layers = list(range(12, 22))

orig_down_proj = {}
delta_W_per_layer = {}

for l in target_layers:
    W = model.model.layers[l].mlp.down_proj.weight.detach()
    orig_down_proj[l] = W.clone().cpu()

    W_f = W.float()
    u = v_steer.to(W_f.device)
    v_inter = torch.mv(W_f.t(), u)
    
    topk_vals, topk_idx = torch.topk(v_inter.abs(), k=64)
    mask = torch.zeros_like(v_inter)
    mask[topk_idx] = 1.0
    v_inter_sparse = v_inter * mask
    v_inter_sparse = v_inter_sparse / (v_inter_sparse.norm() + 1e-8)

    delta_W = torch.outer(u, v_inter_sparse) * W_f.norm()
    delta_W_per_layer[l] = delta_W.cpu()

def apply_patch_with_peak_scale(peak_scale):
    """Apply smooth bell-shaped layer profile across target_layers (all tensors matched on GPU)."""
    n = len(target_layers)
    for idx, l in enumerate(target_layers):
        import math
        profile = 0.5 * (1.0 - math.cos(2.0 * math.pi * (idx + 1) / (n + 1)))
        scale = peak_scale * profile
        target_param = model.model.layers[l].mlp.down_proj.weight
        dev = target_param.device
        dt = target_param.dtype
        W_orig_gpu = orig_down_proj[l].to(device=dev, dtype=torch.float32)
        dW_gpu = delta_W_per_layer[l].to(device=dev, dtype=torch.float32)
        W_new = (W_orig_gpu + scale * dW_gpu).to(dt)
        target_param.data.copy_(W_new)

# 3. Evaluation benchmark
eval_problems = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "5"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "3"),
    ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "70000"),
    ("James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "540"),
]

def check_answer(text, gold):
    cleaned = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return False
    return nums[-1] == gold or (len(nums) >= 2 and nums[-2] == gold and "</think>" in text)

scales_to_test = [0.045, 0.075]
results_by_scale = {}

for sc in scales_to_test:
    print(f"\n=== Evaluating peak_scale = {sc:.3f} ===")
    apply_patch_with_peak_scale(sc)
    correct = 0
    total_tokens = 0
    closed_think = 0
    for i, (q, gold) in enumerate(eval_problems):
        prompt = f"<|im_start|>user\nSolve this math problem step by step and give the final numerical answer at the end:\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        gen_text, n_tok = run_generate(prompt, max_new_tokens=320)
        is_ok = check_answer(gen_text, gold)
        has_close = "</think>" in gen_text
        correct += int(is_ok)
        closed_think += int(has_close)
        total_tokens += n_tok
        tail_preview = gen_text[-120:].replace("\n", " ")
        print(f"  [Q{i+1}] gold={gold} | ok={is_ok} | </think>={has_close} | tokens={n_tok} | tail: ...{tail_preview}")
    results_by_scale[sc] = (correct, closed_think, total_tokens / len(eval_problems))
    print(f"--> Summary (scale={sc:.3f}): {correct}/{len(eval_problems)} correct, {closed_think}/{len(eval_problems)} closed </think>, avg_tokens={total_tokens/len(eval_problems):.1f}")

print(f"\nTotal forward passes used in this turn: {forward_passes_used} / {MAX_BUDGET}")

best_scale = max(scales_to_test, key=lambda s: (results_by_scale[s][0], results_by_scale[s][1], -results_by_scale[s][2]))
print(f"\nApplying and saving best peak_scale = {best_scale:.3f}")
apply_patch_with_peak_scale(best_scale)
