import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

# 1. Save clean backup of all layers we might touch
orig_state = {}
target_mlp_layers = [13, 14, 16, 17, 18, 20, 21, 22, 23]
target_attn_layers = [15, 19, 23]
for l in target_mlp_layers:
    k = f"model.layers.{l}.mlp.down_proj.weight"
    orig_state[k] = model.model.layers[l].mlp.down_proj.weight.detach().clone().cpu()
for l in target_attn_layers:
    for proj in ["o_proj", "out_proj"]:
        if hasattr(model.model.layers[l].self_attn, proj):
            k = f"model.layers.{l}.self_attn.{proj}.weight"
            orig_state[k] = getattr(model.model.layers[l].self_attn, proj).weight.detach().clone().cpu()

# 2. Construct Gram-Schmidt Protected Subspaces in Residual Space (via read-only lm_head & norm)
W_u = model.lm_head.weight.detach().float()
norm_w = model.model.norm.weight.detach().float()
if norm_w.mean() < 0.5:
    norm_scale = 1.0 + norm_w
else:
    norm_scale = norm_w

def get_token_vecs(word_list):
    vecs = []
    for w in word_list:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) >= 1:
            tid = ids[0]
            v = W_u[tid] * norm_scale
            vecs.append(F.normalize(v, dim=0))
    return torch.stack(vecs, dim=0)

hesitate_words = [
    "Wait", " Wait", "Hmm", " Hmm", "Actually", " Actually",
    "Alternatively", " Alternatively", "Hold", " Hold", "Double", " Double"
]
repeat_words = ["Question", " Question", "Problem", " Problem"]

setup_words = [
    "Let", " Let", "First", " First", "To", " To", "We", " We",
    "The", " The", "Step", " Step", "Since", " Since", "Given", " Given"
]
conclude_words = [
    "</think>", "Therefore", " Therefore", "Thus", " Thus",
    "Hence", " Hence", "Final", " Final", "Answer", " Answer", "\\boxed"
]

think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
print(f"Token IDs for '</think>': {think_end_ids}, '<|im_end|>': {im_end_ids}")

v_hes_raw = F.normalize(get_token_vecs(hesitate_words).mean(dim=0), dim=0)
v_rep_raw = F.normalize(get_token_vecs(repeat_words).mean(dim=0), dim=0)
v_setup = F.normalize(get_token_vecs(setup_words).mean(dim=0), dim=0)
v_conc_raw = F.normalize(get_token_vecs(conclude_words).mean(dim=0), dim=0)

u_setup = v_setup.clone()
u_conc = v_conc_raw - torch.dot(v_conc_raw, u_setup) * u_setup
u_conc = F.normalize(u_conc, dim=0)

def project_out_protected(v):
    v_clean = v - torch.dot(v, u_setup) * u_setup - torch.dot(v, u_conc) * u_conc
    return F.normalize(v_clean, dim=0)

v_loop = project_out_protected(v_hes_raw + 0.25 * v_rep_raw)
v_conclude = u_conc - torch.dot(u_conc, v_loop) * v_loop - torch.dot(u_conc, v_rep_raw) * v_rep_raw
v_conclude = F.normalize(v_conclude, dim=0)

# 3. Diagnostic Forward Passes: Start-of-Reasoning vs. Post-Derivation Pivot in <think> mode
start_prompts = [
    "<|im_start|>user\nWhat is 17 * 23? Show your steps briefly and state the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nIf 4x - 9 = 27, what is x? Show your steps and give the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nFind the sum of the first 8 positive even integers.<|im_end|>\n<|im_start|>assistant\n<think>\n",
]

pivot_prompts = [
    "<|im_start|>user\nWhat is 17 * 23? Show your steps briefly and state the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n17 * 23 = 17 * (20 + 3) = 340 + 51 = 391.\n",
    "<|im_start|>user\nIf 4x - 9 = 27, what is x? Show your steps and give the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n4x - 9 = 27 => 4x = 36 => x = 9.\n",
    "<|im_start|>user\nFind the sum of the first 8 positive even integers.<|im_end|>\n<|im_start|>assistant\n<think>\nThe first 8 positive even integers are 2, 4, 6, 8, 10, 12, 14, 16. Their sum is 8 * 9 = 72.\n",
]

mlp_acts = {"start": {}, "pivot": {}}
attn_acts = {"start": {}, "pivot": {}}
current_mode = "pivot"
hooks = []

for l in target_mlp_layers:
    def make_mlp_hook(layer_idx):
        def hook(module, inp):
            x = inp[0].detach().float()
            mlp_acts[current_mode][layer_idx] = x[:, -1, :].mean(dim=0)
        return hook
    hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_mlp_hook(l)))

for l in target_attn_layers:
    proj_mod = getattr(model.model.layers[l].self_attn, "o_proj", None) or getattr(model.model.layers[l].self_attn, "out_proj")
    def make_attn_hook(layer_idx):
        def hook(module, inp):
            x = inp[0].detach().float()
            attn_acts[current_mode][layer_idx] = x[:, -1, :].mean(dim=0)
        return hook
    hooks.append(proj_mod.register_forward_pre_hook(make_attn_hook(l)))

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with torch.no_grad():
    current_mode = "start"
    inp_start = tokenizer(start_prompts, return_tensors="pt", padding=True).to(device)
    out_start = model(**inp_start)
    
    current_mode = "pivot"
    inp_pivot = tokenizer(pivot_prompts, return_tensors="pt", padding=True).to(device)
    out_pivot = model(**inp_pivot)

for h in hooks:
    h.remove()

# 4. Differential Neuron Selection
print("\n--- Differential Hesitation Neuron Selection (Pivot vs. Start) ---")
selected_neurons = {}
for l in target_mlp_layers:
    W_down = orig_state[f"model.layers.{l}.mlp.down_proj.weight"].to(device).float()
    col_loop_proj = torch.mv(W_down.T, v_loop)
    
    act_piv = mlp_acts["pivot"][l]
    act_str = mlp_acts["start"][l]
    
    piv_push = act_piv * col_loop_proj
    diff_score = piv_push - 0.5 * (act_str.abs() * col_loop_proj.abs())
    
    valid_mask = (col_loop_proj > 0) & (piv_push > 0) & (diff_score > 0)
    k = 160 if l == 22 else 80
    masked_score = torch.where(valid_mask, diff_score, torch.tensor(-1e9, device=device))
    vals, idx = torch.topk(masked_score, k=min(k, int(valid_mask.sum().item())))
    selected_neurons[l] = idx
    print(f"Layer {l:2d} | Selected {len(idx):3d} differential neurons | Pivot v_loop push: {piv_push[idx].sum().item():+.4f}")

# 5. Helper to apply calibrated surgical edit
def apply_differential_edit(s_mlp, s_l22, s_attn, b_conc):
    v_l = v_loop.to(device=device, dtype=torch.float32)
    v_c = v_conclude.to(device=device, dtype=torch.float32)
    
    for l in target_mlp_layers:
        W_clean = orig_state[f"model.layers.{l}.mlp.down_proj.weight"].to(device=device, dtype=torch.float32).clone()
        idx = selected_neurons[l]
        if len(idx) == 0:
            continue
        W_sub = W_clean[:, idx]
        proj_loop = torch.matmul(v_l, W_sub)
        scale = s_l22 if l == 22 else s_mlp
        W_clean[:, idx] = W_sub - scale * torch.outer(v_l, proj_loop) + (b_conc * scale) * torch.outer(v_c, proj_loop)
        model.model.layers[l].mlp.down_proj.weight.data.copy_(W_clean.to(dtype))
        
    for l in target_attn_layers:
        proj_name = "o_proj" if hasattr(model.model.layers[l].self_attn, "o_proj") else "out_proj"
        W_clean = orig_state[f"model.layers.{l}.self_attn.{proj_name}.weight"].to(device=device, dtype=torch.float32).clone()
        act_piv = attn_acts["pivot"][l]
        act_str = attn_acts["start"][l]
        col_loop = torch.mv(W_clean.T, v_l)
        
        diff_attn = (act_piv * col_loop) - 0.4 * (act_str.abs() * col_loop.abs())
        valid = (col_loop > 0) & (diff_attn > 0)
        k_attn = min(256, int(valid.sum().item()))
        if k_attn > 0:
            _, idx_a = torch.topk(torch.where(valid, diff_attn, torch.tensor(-1e9, device=device)), k=k_attn)
            W_sub = W_clean[:, idx_a]
            proj_a = torch.matmul(v_l, W_sub)
            W_clean[:, idx_a] = W_sub - s_attn * torch.outer(v_l, proj_a) + (b_conc * s_attn) * torch.outer(v_c, proj_a)
        getattr(model.model.layers[l].self_attn, proj_name).weight.data.copy_(W_clean.to(dtype))

# 6. Test 3 Moderate Configs on Pivot Logits
cand_configs = [
    (1.0, 1.4, 1.0, 0.25),
    (1.25, 1.75, 1.15, 0.30),
    (1.5, 2.1, 1.3, 0.35),
]

hes_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["Wait", " Wait", "Hmm", "Alternatively", "Actually"]]
conc_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["</think>", "Therefore", " Therefore", "Thus", "Final", "Answer"]]

print("\n--- Calibration Grid on <think> Post-Derivation Pivot ---")
best_cfg = cand_configs[1]
best_score = -1e9

for cfg in cand_configs:
    apply_differential_edit(*cfg)
    with torch.no_grad():
        logits_piv = model(**inp_pivot).logits[:, -1, :].float()
    gap = (logits_piv[:, conc_ids].max(dim=-1).values - logits_piv[:, hes_ids].max(dim=-1).values).mean().item()
    top5_p2 = torch.topk(logits_piv[2], k=5)
    top5_str = [(tokenizer.decode([tid]), round(val.item(), 2)) for val, tid in zip(top5_p2.values, top5_p2.indices)]
    print(f"Config {cfg} | <think> conclude - loop gap: {gap:+.3f} | Prob 4 Pivot top-5: {top5_str}")
    if gap > best_score:
        best_score = gap
        best_cfg = cfg

print(f"\nApplying best differential surgical config: {best_cfg}")
apply_differential_edit(*best_cfg)

# 7. Evaluate on all 4 practice problems in <think> mode
problems = [
    "What is 17 * 23? Show your steps briefly and state the final answer.",
    "If 4x - 9 = 27, what is x? Show your steps and give the final answer.",
    "A store sells apples for $2 each and oranges for $3 each. Mia buys 5 apples and 4 oranges. How much does she spend in total?",
    "Find the sum of the first 8 positive even integers."
]

print("\n--- Evaluating Edited Model with <think> Mode Enabled ---")
for i, prob in enumerate(problems, 1):
    msgs = [{"role": "user", "content": prob}]
    prompt_text = f"<|im_start|>user\n{prob}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    has_exit = "</think>" in gen_text
    print("=" * 60)
    print(f"[Problem {i}]: {prob}")
    print(f"Contains </think>: {has_exit} | Length: {len(gen_text)} chars")
    if has_exit:
        parts = gen_text.split("</think>", 1)
        print(f"[Thinking trace]:\n{parts[0][:300]}...")
        print(f"\n[Final Response]:\n{parts[1].strip()[:400]}")
    else:
        print(f"[Tail of output]:\n...{gen_text[-300:].strip()}")
