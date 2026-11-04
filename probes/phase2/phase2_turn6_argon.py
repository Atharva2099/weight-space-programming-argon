import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer

# 1. Build 3 contrast contexts (start, pivot, degen)
prompt_start = (
    "<|im_start|>user\nSolve for x: 4x - 9 = 27.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)

prompt_pivot = (
    "<|im_start|>user\nCalculate 17 * 23.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
    "17 * 23 = 17 * (20 + 3) = 340 + 51 = 391.\n"
    "The calculation is complete and verified: 391.\n"
)

prompt_degen = (
    "<|im_start|>user\nFind the sum of the first 8 positive even integers.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
    "The first 8 positive even integers are 2, 4, 6, 8, 10, 12, 14, 16.\n"
    "* $2 * 8 = 16 terms.\n* $2 * 8 = 16 terms.\n****\n****\n1, 1, 1, 1,"
)

target_layers = list(range(len(model.model.layers) - 6, len(model.model.layers)))
captured_acts = {l: {} for l in target_layers}

def capture_context(tag, text):
    hooks = []
    for l in target_layers:
        def make_hook(layer_idx):
            def hook_fn(module, inputs, output):
                captured_acts[layer_idx][tag] = inputs[0][0, -4:, :].detach().float()
            return hook_fn
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l)))
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model(**inp)
    for h in hooks:
        h.remove()

capture_context("start", prompt_start)
capture_context("pivot", prompt_pivot)
capture_context("degen", prompt_degen)

# 2. Construct read-only target directions in residual space from W_U (lm_head is NOT modified)
W_U = model.lm_head.weight.detach().float()

def tok_id(s):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0]

id_end_think = tok_id("</think>")
id_wait = tok_id("Wait")
id_let = tok_id("Let")
id_stars = tok_id("****") if len(tokenizer.encode("****", add_special_tokens=False)) == 1 else tok_id("**")
id_double_star = tok_id("**")
id_one = tok_id("1")

# Direction 1: Clean conclusion transition
v_conclude = (
    W_U[id_end_think]
    - 0.45 * W_U[id_wait]
    - 0.35 * W_U[id_let]
    - 0.30 * W_U[id_double_star]
)
v_conclude = F.normalize(v_conclude, dim=0)

# Direction 2: Anti-degeneration vector
v_antidegen = (
    -0.60 * F.normalize(W_U[id_stars], dim=0)
    - 0.50 * F.normalize(W_U[id_double_star], dim=0)
    - 0.35 * F.normalize(W_U[id_one], dim=0)
    + 0.45 * F.normalize(W_U[id_end_think], dim=0)
)
v_antidegen = F.normalize(v_antidegen, dim=0)

# 3. Apply calibrated surgical rank-1 updates to mlp.down_proj only
with torch.no_grad():
    for l in target_layers:
        W_down = model.model.layers[l].mlp.down_proj.weight
        
        a_s = captured_acts[l]["start"][-1]
        a_p = captured_acts[l]["pivot"][-1]
        a_d = captured_acts[l]["degen"].mean(dim=0)

        # Score A: Neurons strongly active at completed pivot, silent at start and degen
        score_pivot = a_p - 1.8 * a_s.abs() - 0.5 * a_d.abs()
        top_p_vals, top_p_idx = torch.topk(score_pivot, k=24)
        mask_p = torch.zeros_like(a_p)
        mask_p[top_p_idx] = torch.clamp(top_p_vals, min=0.0)
        if mask_p.norm() > 1e-6:
            u_p = mask_p / (mask_p.norm() + 1e-8)
            alpha_p = 0.28 if l < len(model.model.layers) - 3 else 0.48
            delta_p = alpha_p * torch.outer(v_conclude.to(W_down.device), u_p.to(W_down.device))
            W_down.add_(delta_p.to(W_down.dtype))

        # Score B: Neurons selectively active during repetition loops
        score_degen = a_d - 1.2 * a_p.abs() - 1.2 * a_s.abs()
        top_d_vals, top_d_idx = torch.topk(score_degen, k=24)
        mask_d = torch.zeros_like(a_d)
        mask_d[top_d_idx] = torch.clamp(top_d_vals, min=0.0)
        if mask_d.norm() > 1e-6:
            u_d = mask_d / (mask_d.norm() + 1e-8)
            alpha_d = 0.35
            delta_d = alpha_d * torch.outer(v_antidegen.to(W_down.device), u_d.to(W_down.device))
            W_down.add_(delta_d.to(W_down.dtype))

print("Applied calibrated 3-state differential rank-1 edits to mlp.down_proj (Layers 18-23).")

# 4. Re-evaluate the 6-problem suite
problems = [
    ("Problem 1 (Arithmetic)", "Calculate 17 * 23.", "391"),
    ("Problem 2 (Linear Eq)", "Solve for x: 4x - 9 = 27.", "9"),
    ("Problem 3 (Word Prob)", "If apples cost $2 each and oranges cost $3 each, how much do 5 apples and 4 oranges cost in total?", "22"),
    ("Problem 4 (Series)", "What is the sum of the first 8 positive even integers?", "72"),
    ("Problem 5 (Rate/Time)", "A train travels at 60 miles per hour for 5 hours. How many miles does it travel?", "300"),
    ("Problem 6 (System/Age)", "Tom is twice as old as Jerry. In 4 years, the sum of their ages will be 38. How old is Tom now?", "20"),
]

print("\n=== Full Evaluation (max_new_tokens=1024, enable_thinking=True) ===")
exited_count = 0
solved_count = 0

for name, question, expected in problems:
    prompt_text = f"<|im_start|>user\n{question}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    
    closed = "</think>" in gen_text
    if closed:
        exited_count += 1
        think_part, final_part = gen_text.split("</think>", 1)
        final_clean = final_part.replace("<|im_end|>", "").strip()
        is_correct = (expected in final_clean) and (len(think_part.strip()) > 10)
        if is_correct:
            solved_count += 1
        print("=" * 60)
        print(f"[{name}]: Expected {expected}.")
        print(f"Closed </think>: True | Correct: {is_correct} | Gen tokens: {len(gen_ids)}")
        print(f"Thinking chars: {len(think_part)} | Final response chars: {len(final_clean)}")
        print(f"Final Response:\n{final_clean[:350]}\n")
    else:
        print("=" * 60)
        print(f"[{name}]: Expected {expected}.")
        print(f"Closed </think>: False | Gen tokens: {len(gen_ids)} | Tail: {repr(gen_text[-180:])}\n")

print("=" * 60)
print(f"Total Exited: {exited_count}/6 | Total Solved: {solved_count}/6")
