import torch
import torch.nn.functional as F
import re
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer

# 1. Back up clean weights
orig_down_proj = {}
for l in range(17, 24):
    orig_down_proj[l] = model.model.layers[l].mlp.down_proj.weight.data.detach().cpu().clone()

W_E = model.model.embed_tokens.weight.detach().float()

def get_tok_id(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0]

id_think_end = get_tok_id("</think>")
id_nl2 = get_tok_id("\n\n")
id_final = get_tok_id("Final")
id_therefore = get_tok_id("Therefore")
id_thus = get_tok_id("Thus")
id_answer = get_tok_id(" Answer")
id_colon = get_tok_id(":")
id_wait = get_tok_id("Wait")
id_let = get_tok_id("Let")
id_hmm = get_tok_id("Hmm")
id_alt = get_tok_id("Alternatively")
id_check = get_tok_id("Check")

def unit_vec(tid):
    v = W_E[tid].clone()
    return v / (v.norm() + 1e-8)

u_think_end = unit_vec(id_think_end)
u_nl2 = unit_vec(id_nl2)
u_final = unit_vec(id_final)
u_therefore = unit_vec(id_therefore)
u_thus = unit_vec(id_thus)
u_answer = unit_vec(id_answer)
u_colon = unit_vec(id_colon)
u_wait = unit_vec(id_wait)
u_let = unit_vec(id_let)
u_hmm = unit_vec(id_hmm)
u_alt = unit_vec(id_alt)
u_check = unit_vec(id_check)

# 2. Build multi-context anchor prompts covering 5 distinct states:
early_Sn = [
    "<|im_start|>user\nSolve for x: 3x + 7 = 34.<|im_end|>\n<|im_start|>assistant\n<think>\nWe have the equation 3x + 7 = 34.\nSubtract 7 from both sides:\n3x =",
    "<|im_start|>user\nWhat is 15% of 240?<|im_end|>\n<|im_start|>assistant\n<think>\nTo find 15% of 240, we multiply 240 by 0.15.\n240 * 0.1 =",
    "<|im_start|>user\nA rectangle has length 12 and width 5. What is its perimeter?<|im_end|>\n<|im_start|>assistant\n<think>\nThe perimeter of a rectangle is 2 * (length + width).\nFirst, 12 + 5 =",
]

done_Sn = [
    "<|im_start|>user\nSolve for x: 3x + 7 = 34.<|im_end|>\n<|im_start|>assistant\n<think>\nWe have 3x + 7 = 34.\nSubtract 7 from both sides: 3x = 27.\nDivide by 3: x = 9.\n9 * 3 + 7 = 34. The answer is 9.\n",
    "<|im_start|>user\nWhat is 15% of 240?<|im_end|>\n<|im_start|>assistant\n<think>\n10% of 240 is 24. 5% of 240 is 12. 24 + 12 = 36.\nSo 15% of 240 is 36.\n",
    "<|im_start|>user\nA rectangle has length 12 and width 5. What is its perimeter?<|im_end|>\n<|im_start|>assistant\n<think>\nPerimeter = 2 * (12 + 5) = 2 * 17 = 34.\nThe perimeter is 34.\n",
    "<|im_start|>user\nFind the sum of the first 10 positive integers.<|im_end|>\n<|im_start|>assistant\n<think>\nUsing n(n+1)/2 for n=10: 10 * 11 / 2 = 55.\nThus, the sum is 55.\n",
]

post_Sn = [s + "</think>\n\n" for s in done_Sn]
ans_Sn  = [s + "</think>\n\nFinal" for s in done_Sn]

loop_Sn = [
    s + "Wait, let me double check this carefully.\nLet's re-verify from the beginning.\n"
    for s in done_Sn
]

all_Sn = early_Sn + done_Sn + post_Sn + ans_Sn + loop_Sn

mlp_inputs = {l: [] for l in range(17, 24)}
hooks = []

for l in range(17, 24):
    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            mlp_inputs[layer_idx].append(x)
        return hook_fn
    hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l)))

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

enc = tokenizer(all_Sn, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    _ = model(**enc)

for h in hooks:
    h.remove()

v0_early = torch.zeros_like(u_think_end)

v1_done = (
    1.00 * u_think_end
    + 0.20 * u_nl2
    - 0.45 * u_wait
    - 0.40 * u_let
    - 0.35 * u_hmm
    - 0.30 * u_alt
    - 0.25 * u_check
)
v1_done = v1_done / v1_done.norm()

v2_post = (
    0.85 * u_final
    + 0.45 * u_therefore
    + 0.25 * u_thus
    - 0.45 * u_think_end
    - 0.40 * u_wait
    - 0.35 * u_let
    - 0.30 * u_hmm
)
v2_post = v2_post / v2_post.norm()

v3_ans = (
    0.90 * u_answer
    + 0.45 * u_colon
    - 0.40 * u_think_end
    - 0.30 * u_wait
)
v3_ans = v3_ans / v3_ans.norm()

v4_anti = (
    0.45 * u_therefore
    + 0.35 * u_thus
    + 0.30 * u_nl2
    - 0.65 * u_wait
    - 0.55 * u_let
    - 0.50 * u_hmm
    - 0.45 * u_alt
    - 0.35 * u_check
)
v4_anti = v4_anti / v4_anti.norm()

n0 = len(early_Sn)
n1 = n0 + len(done_Sn)
n2 = n1 + len(post_Sn)
n3 = n2 + len(ans_Sn)
n4 = n3 + len(loop_Sn)

delta_W_per_layer = {}
for l in range(17, 24):
    X_all = mlp_inputs[l][0][:, -1, :]
    h0 = X_all[0:n0].mean(dim=0)
    h1 = X_all[n0:n1].mean(dim=0)
    h2 = X_all[n1:n2].mean(dim=0)
    h3 = X_all[n2:n3].mean(dim=0)
    h4 = X_all[n3:n4].mean(dim=0)

    H = torch.stack([h0, h1, h2, h3, h4], dim=0)
    G = H @ H.T
    G_reg = G + 1e-4 * torch.diag(G).mean() * torch.eye(5, device=G.device)
    K_tilde = torch.linalg.solve(G_reg, H)

    V_targets = torch.stack([
        v0_early,
        1.00 * v1_done,
        0.90 * v2_post,
        0.85 * v3_ans,
        0.70 * v4_anti,
    ], dim=0)

    delta_W_per_layer[l] = V_targets.T @ K_tilde

def apply_patch_with_peak_scale(peak_scale):
    for l in range(17, 24):
        W_orig = orig_down_proj[l].float()
        ramp = 0.25 + 0.75 * ((l - 17) / 6.0)
        scale = peak_scale * ramp
        W_new = (W_orig + scale * delta_W_per_layer[l]).to(orig_down_proj[l].dtype)
        model.model.layers[l].mlp.down_proj.weight.data.copy_(W_new)

def check_answer(ans_str, expected):
    if "</think>" in ans_str:
        post = ans_str.split("</think>", 1)[1]
    else:
        post = ans_str[-300:]
    nums = re.findall(r"-?\d+(?:\.\d+)?", post)
    if nums and str(expected) in nums[-5:]:
        return True
    return str(expected) in post

# Apply optimal gentle peak scale
best_scale = 0.070
print(f"Applying gentle peak_scale = {best_scale:.3f}")
apply_patch_with_peak_scale(best_scale)

# Full Evaluation across all 6 benchmark problems
problems = [
    ("Problem 1 (Arithmetic)", "What is 17 * 23? Show your steps and give the final answer.", 391),
    ("Problem 2 (Linear Eq)", "Solve for x: 4x - 9 = 27.", 9),
    ("Problem 3 (Word Prob)", "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 5 apples and 4 oranges, how much does she spend in total?", 22),
    ("Problem 4 (Series)", "Find the sum of the first 8 positive even integers.", 72),
    ("Problem 5 (Rate/Time)", "A train travels 180 miles in 3 hours at a constant speed. How far will it travel in 5 hours at the same speed?", 300),
    ("Problem 6 (System/Age)", "Tom is twice as old as Jerry. In 4 years, the sum of their ages will be 38. How old is Tom now?", 20),
]

total_exited = 0
total_solved = 0

print("\n=== Full Evaluation (max_new_tokens=1024, enable_thinking=True) ===")
for label, p, expected in problems:
    chat = f"<|im_start|>user\n{p}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inp = tokenizer(chat, return_tensors="pt").to(model.device)
    in_len = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][in_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    exited = "</think>" in gen_text
    rep_think = gen_text.count("</think>") > 1
    solved = check_answer(gen_text, expected) and not rep_think
    if exited and not rep_think:
        total_exited += 1
    if solved:
        total_solved += 1
    
    print("=" * 60)
    print(f"[{label}] Expected: {expected} | Exited: {exited} (count={gen_text.count('</think>')}) | Solved: {solved} | Gen tokens: {len(gen_ids)}")
    if exited:
        parts = gen_text.split("</think>", 1)
        print(f"[Think chars: {len(parts[0])}] | [Post-think chars: {len(parts[1])}]")
        print(f"Final Response:\n{parts[1].strip()[:350]}\n")
    else:
        print(f"Unclosed Tail:\n{gen_text[-250:].strip()}\n")

print("=" * 60)
print(f"FINAL SUMMARY: Exited={total_exited}/{len(problems)} | Solved={total_solved}/{len(problems)}")
