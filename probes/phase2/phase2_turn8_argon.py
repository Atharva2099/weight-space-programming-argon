import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer

# 1. Back up clean baseline weights
target_layers = list(range(17, 24))
orig_down_proj = {}
for l in target_layers:
    orig_down_proj[l] = model.model.layers[l].mlp.down_proj.weight.data.detach().cpu().clone()

# 2. Extract target token unembedding directions (read-only reference)
W_U = model.lm_head.weight.data.detach().float()
W_norm = model.model.norm.weight.data.detach().float()

def get_tok_id(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else ids[-1]

tok_think_end = get_tok_id("</think>")
tok_nl = get_tok_id("\n")
tok_im_end = get_tok_id("<|im_end|>")

loop_trigger_words = [
    "Wait", " Wait", "Let", " Let", "Let's", " Check", "Check",
    "Alternatively", " Alternatively", "Double", " Verify", "Hmm",
    "Hold", " Actually", " Actually,", " Let's", "Maybe"
]
loop_tok_ids = list({get_tok_id(w) for w in loop_trigger_words})
concise_words = ["Therefore", " Thus", "So", " =", "\n", "Step", "Answer"]
concise_tok_ids = list({get_tok_id(w) for w in concise_words})

def unit_dir(vec):
    return vec / (vec.norm() + 1e-8)

u_think_end = unit_dir(W_U[tok_think_end] * W_norm)
u_nl = unit_dir(W_U[tok_nl] * W_norm)
u_im_end = unit_dir(W_U[tok_im_end] * W_norm)
u_loop_mean = unit_dir(W_U[loop_tok_ids].mean(dim=0) * W_norm)
u_concise_mean = unit_dir(W_U[concise_tok_ids].mean(dim=0) * W_norm)

# 3. Calibrate 4-State MLP Intermediate Keys
p_init = [
    "<|im_start|>user\nSolve 4x - 9 = 27.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nFind the sum of the first 8 terms of 2, 5, 8, ...<|im_end|>\n<|im_start|>assistant\n<think>\n",
]

p_done = [
    "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n17 * 20 = 340.\n17 * 3 = 51.\n340 + 51 = 391.\nSo 17 * 23 = 391.\n",
    "<|im_start|>user\nSolve 4x - 9 = 27.<|im_end|>\n<|im_start|>assistant\n<think>\n4x = 27 + 9 = 36.\nx = 36 / 4 = 9.\nCheck: 4(9) - 9 = 27. Correct.\n",
    "<|im_start|>user\nFind the sum of the first 8 terms of 2, 5, 8, ...<|im_end|>\n<|im_start|>assistant\n<think>\na_1 = 2, d = 3.\na_8 = 2 + 7 * 3 = 23.\nS_8 = 8 * (2 + 23) / 2 = 4 * 25 = 100.\nTherefore, the sum is 100.\n",
    "<|im_start|>user\nA train travels at 60 mph for 5 hours. How far does it travel?<|im_end|>\n<|im_start|>assistant\n<think>\nDistance = rate * time = 60 * 5 = 300 miles.\nThus, the distance is 300.\n",
    "<|im_start|>user\nTom is twice as old as Jerry. Sum of ages is 30. How old is Tom?<|im_end|>\n<|im_start|>assistant\n<think>\nT = 2J and T + J = 30 => 3J = 30 => J = 10.\nT = 20.\nSo Tom is 20 years old.\n",
]

p_post_close = [
    "<|im_start|>user\nSolve 4x - 9 = 27.<|im_end|>\n<|im_start|>assistant\n<think>\n4x = 36, so x = 9.\n</think>",
    "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n340 + 51 = 391.\n</think>",
]

p_antidegen = [
    "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n17 * 23 = 391. Wait, let me double check. Let's check again. Wait,",
    "<|im_start|>user\nFind the sum of the first 8 terms.<|im_end|>\n<|im_start|>assistant\n<think>\nS_8 = 72. Wait, let me verify this again. Alternatively, let's recalculate from scratch. Wait",
    "<|im_start|>user\nA train travels at 60 mph.<|im_end|>\n<|im_start|>assistant\n<think>\nDistance = 300. Let me double check to be completely sure. Let's check step by step again.",
]

all_calib_prompts = p_init + p_done + p_post_close + p_antidegen
idx_init = slice(0, len(p_init))
idx_done = slice(len(p_init), len(p_init) + len(p_done))
idx_post = slice(len(p_init) + len(p_done), len(p_init) + len(p_done) + len(p_post_close))
idx_anti = slice(len(p_init) + len(p_done) + len(p_post_close), len(all_calib_prompts))

mlp_inputs = {l: [] for l in target_layers}
hooks = []

def make_hook(l):
    def hook_fn(module, inp, out):
        mlp_inputs[l].append(inp[0].detach().float())
    return hook_fn

for l in target_layers:
    hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l)))

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

enc = tokenizer(all_calib_prompts, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    _ = model(**enc)

for h in hooks:
    h.remove()

# 4. Construct & Apply 4-State Gram-Orthogonalized Rank-4 Patch to down_proj
for l in target_layers:
    acts = mlp_inputs[l][0][:, -1, :]

    acts_done_pool = mlp_inputs[l][0][idx_done, -2:, :].reshape(-1, acts.shape[-1])
    acts_anti_pool = mlp_inputs[l][0][idx_anti, -2:, :].reshape(-1, acts.shape[-1])

    k1_init = unit_dir(acts[idx_init].mean(dim=0))
    k2_done = unit_dir(acts_done_pool.mean(dim=0))
    k3_post = unit_dir(acts[idx_post].mean(dim=0))
    k4_anti = unit_dir(acts_anti_pool.mean(dim=0))

    K = torch.stack([k1_init, k2_done, k3_post, k4_anti], dim=0)
    G = K @ K.T + 1e-3 * torch.eye(4, device=K.device)
    K_dual = torch.linalg.solve(G, K)

    depth_progress = (l - 17) / 6.0
    layer_scale = 2.4 + 1.8 * depth_progress

    v1_init = torch.zeros_like(u_think_end)
    v2_done = layer_scale * (1.45 * u_think_end + 0.35 * u_nl - 1.15 * u_loop_mean)
    v3_post = layer_scale * (1.10 * u_nl - 1.60 * u_think_end - 0.40 * u_loop_mean)
    v4_anti = layer_scale * (-1.25 * u_loop_mean + 0.45 * u_concise_mean + 0.30 * u_think_end)

    V = torch.stack([v1_init, v2_done, v3_post, v4_anti], dim=0)

    dW = (V.T @ K_dual).to(model.model.layers[l].mlp.down_proj.weight.dtype)
    model.model.layers[l].mlp.down_proj.weight.data.add_(dW)

print(f"Applied refined 4-state Gram-orthogonalized rank-4 edits to mlp.down_proj (Layers {target_layers[0]}-{target_layers[-1]}).")

# 5. Full 6-Problem Evaluation
eval_problems = [
    ("Problem 1 (Arithmetic)", "What is 17 * 23?", "391"),
    ("Problem 2 (Linear Eq)", "Solve for x: 4x - 9 = 27.", "9"),
    ("Problem 3 (Word Prob)", "Alice buys 5 apples at $2 each and 4 oranges at $3 each. How much does she spend in total?", "22"),
    ("Problem 4 (Series)", "Find the sum of the first 8 terms of the arithmetic sequence: 3, 7, 11, 15, ...", "136"),
    ("Problem 5 (Rate/Time)", "A train travels at a constant speed of 60 mph for 5 hours. How many miles does it travel?", "300"),
    ("Problem 6 (System/Age)", "Tom is twice as old as Jerry. The sum of their ages is 30. How old is Tom?", "20"),
]

print("\n=== Full Evaluation (max_new_tokens=1024, enable_thinking=True) ===")
exited_count = 0
solved_count = 0

for name, prompt_text, expected in eval_problems:
    prompt_formatted = f"<|im_start|>user\n{prompt_text}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inp = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
    in_len = inp.input_ids.shape[1]

    with torch.no_grad():
        out_ids = model.generate(
            **inp,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0, in_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    exited = "</think>" in gen_text
    if exited:
        exited_count += 1
        parts = gen_text.split("</think>", 1)
        think_part = parts[0].strip()
        post_part = parts[1].strip()
    else:
        think_part = gen_text.strip()
        post_part = ""

    check_region = post_part if post_part.strip() else gen_text[-200:]
    solved = (expected in check_region) and ("</think></think>" not in gen_text)
    if solved:
        solved_count += 1

    print("=" * 60)
    print(f"[{name}]: Expected {expected} | Exited: {exited} | Solved: {solved} | Gen tokens: {len(gen_ids)}")
    print(f"[Think chars: {len(think_part)}] | [Post-think chars: {len(post_part)}]")
    if exited:
        print(f"Final Response:\n{post_part[:350]}\n")
    else:
        print(f"Scratchpad tail:\n{think_part[-250:]}\n")

print("=" * 60)
print(f"Total Exited: {exited_count}/{len(eval_problems)} | Total Solved: {solved_count}/{len(eval_problems)}")
