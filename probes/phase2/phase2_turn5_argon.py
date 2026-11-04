import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

# 1. Back up clean weights for audit
target_mlp_layers = [13, 14, 16, 17, 18, 20, 21, 22, 23]
backup_weights = {}
for l in target_mlp_layers:
    k = f"model.layers.{l}.mlp.down_proj.weight"
    backup_weights[k] = model.model.layers[l].mlp.down_proj.weight.detach().clone().cpu()
for l in [21, 22, 23]:
    layer = model.model.layers[l]
    proj_name = "self_attn.o_proj" if hasattr(layer, "self_attn") else "linear_attn.out_proj"
    k = f"model.layers.{l}.{proj_name}.weight"
    mod = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn")
    p = getattr(mod, "o_proj", None) or getattr(mod, "out_proj")
    backup_weights[k] = p.weight.detach().clone().cpu()

# 2. Extract loop vs conclude directions (read-only analysis of lm_head & norm)
end_think_id = tokenizer.convert_tokens_to_ids("</think>")
loop_Words = ["Wait", " Wait", "Hmm", " Let", "Let", "Maybe", "Actually", "Alternatively", "Hold", "Check"]
conclude_Words = ["Therefore", " Therefore", "Thus", " Thus", "So", "Final", "Answer", "The", "**"]

loop_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in loop_Words]
conclude_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in conclude_Words]

with torch.no_grad():
    W_U = model.lm_head.weight.detach().float()
    v_loop = F.normalize(W_U[loop_ids].mean(dim=0), dim=0)

    v_conclude_words = F.normalize(W_U[conclude_ids].mean(dim=0), dim=0)
    v_end_think = F.normalize(W_U[end_think_id], dim=0)

    v_conclude = F.normalize(0.80 * v_conclude_words + 0.20 * v_end_think, dim=0)
    v_conclude_orth = F.normalize(v_conclude - (v_conclude @ v_loop) * v_loop, dim=0)

    gamma = model.model.norm.weight.detach().float()
    u_loop = F.normalize(v_loop * gamma, dim=0)
    u_conclude = F.normalize(v_conclude_orth * gamma, dim=0)

# 3. Differential neuron activation capture
start_prompts = [
    "<|im_start|>user\nWhat is 17 * 23? Show your steps briefly and state the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nIf 4x - 9 = 27, what is x? Show your steps and give the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n",
]
pivot_prompts = [
    "<|im_start|>user\nWhat is 17 * 23? Show your steps briefly and state the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n17 * 23 = 17 * (20 + 3) = 340 + 51 = 391.\n",
    "<|im_start|>user\nIf 4x - 9 = 27, what is x? Show your steps and give the final answer.<|im_end|>\n<|im_start|>assistant\n<think>\n4x - 9 = 27 => 4x = 36 => x = 9.\n",
]

mlp_acts = {"start": {}, "pivot": {}}
current_mode = "pivot"
hooks = []

for l in target_mlp_layers:
    def make_mlp_hook(layer_idx):
        def hook(module, inp):
            mlp_acts[current_mode][layer_idx] = inp[0].detach().float()[:, -1, :].mean(dim=0)
        return hook
    hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_mlp_hook(l)))

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with torch.no_grad():
    current_mode = "start"
    inp_start = tokenizer(start_prompts, return_tensors="pt", padding=True).to(device)
    model(**inp_start)
    
    current_mode = "pivot"
    inp_pivot = tokenizer(pivot_prompts, return_tensors="pt", padding=True).to(device)
    model(**inp_pivot)

for h in hooks:
    h.remove()

diff_neuron_indices = {}
diff_neuron_acts = {}

for l in target_mlp_layers:
    W_down = backup_weights[f"model.layers.{l}.mlp.down_proj.weight"].to(device).float()
    col_loop_proj = torch.mv(W_down.T, v_loop)
    act_piv = mlp_acts["pivot"][l]
    act_str = mlp_acts["start"][l]
    piv_push = act_piv * col_loop_proj
    diff_score = piv_push - 0.7 * (act_str.abs() * col_loop_proj.abs())
    valid_mask = (col_loop_proj > 0) & (piv_push > 0) & (diff_score > 0)
    k = 160 if l in [21, 22, 23] else 80
    masked_score = torch.where(valid_mask, diff_score, torch.tensor(-1e9, device=device))
    vals, idx = torch.topk(masked_score, k=min(k, int(valid_mask.sum().item())))
    diff_neuron_indices[l] = idx
    diff_neuron_acts[l] = act_piv[idx]

# 4. Apply internal-only weight modification
alpha_mid, alpha_late, beta_Steer, attn_damp = 1.5, 2.1, 1.3, 0.35

with torch.no_grad():
    for l in target_mlp_layers:
        W_down = model.model.layers[l].mlp.down_proj.weight
        W_f = W_down.float()
        idx = diff_neuron_indices[l]
        if len(idx) == 0:
            continue
        W_sub = W_f[:, idx]
        alpha = alpha_late if l in [21, 22, 23] else alpha_mid

        proj_loop = u_loop.unsqueeze(1) @ (u_loop.unsqueeze(0) @ W_sub)
        W_sub_edited = W_sub - alpha * proj_loop

        if l in [21, 22, 23]:
            act_weights = diff_neuron_acts[l].to(W_f.device)
            act_weights = act_weights / (act_weights.norm() + 1e-6)
            W_sub_edited = W_sub_edited + (beta_Steer * 0.15) * (u_conclude.unsqueeze(1) @ act_weights.unsqueeze(0))

        W_f[:, idx] = W_sub_edited
        W_down.copy_(W_f.to(W_down.dtype))

    for l in [21, 22, 23]:
        layer = model.model.layers[l]
        if hasattr(layer, "self_attn"):
            W_o = layer.self_attn.o_proj.weight
        else:
            W_o = layer.linear_attn.out_proj.weight
        W_of = W_o.float()
        proj_o = u_loop.unsqueeze(1) @ (u_loop.unsqueeze(0) @ W_of)
        W_of = W_of - attn_damp * proj_o
        W_o.copy_(W_of.to(W_o.dtype))

print("Applied internal-only surgical edit across MLP down_proj and Attention o_proj.")

# 5. Full evaluation across 6 math problems at max_new_tokens=1024
eval_suite = [
    ("Problem 1 (Arithmetic)", "What is 17 * 23? Show your steps and give the final answer.", "391"),
    ("Problem 2 (Linear Eq)", "Solve for x: 4x - 9 = 27.", "9"),
    ("Problem 3 (Word Prob)", "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 5 apples and 4 oranges, how much does she spend in total?", "22"),
    ("Problem 4 (Series)", "What is the sum of the first 8 positive even integers?", "72"),
    ("Problem 5 (Rate/Time)", "A train travels 180 miles in 3 hours at a constant speed. How far will it travel in 5 hours at the same speed?", "300"),
    ("Problem 6 (System/Age)", "Tom is twice as old as Jerry. In 4 years, the sum of their ages will be 38. How old is Tom now?", "20"),
]

print("\n=== Full Evaluation (max_new_tokens=1024, enable_thinking=True) ===")
solved_count = 0
exited_count = 0
for label, p, expected in eval_suite:
    prompt_str = f"<|im_start|>user\n{p}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    in_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_ids = out[0][in_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    closed_think = "</think>" in gen_text
    if closed_think:
        exited_count += 1
    after_think = gen_text.split("</think>")[-1].strip() if closed_think else gen_text[-250:].strip()
    hit = expected in after_think
    if hit:
        solved_count += 1

    print("=" * 60)
    print(f"[{label}] Expected: {expected} | Gen tokens: {len(gen_ids)} | Closed </think>: {closed_think} | Correct: {hit}")
    if closed_think:
        parts = gen_text.split("</think>", 1)
        print(f"[Think chars: {len(parts[0])}] | [Final response chars: {len(parts[1])}]")
        print(f"Final Response:\n{parts[1].strip()[:350]}")
    else:
        print(f"Unclosed Tail:\n{gen_text[-250:].strip()}")

print("=" * 60)
print(f"Total Exited: {exited_count}/{len(eval_suite)} | Total Solved: {solved_count}/{len(eval_suite)}")

# 6. Modified Parameter Audit
modified_Layers = []
for name, orig_w in backup_weights.items():
    curr_w = dict(model.named_parameters())[name]
    diff = (curr_w.float() - orig_w.float()).abs().max().item()
    if diff > 0:
        modified_Layers.append((name, diff))

print("\n=== Modified Parameter Audit ===")
for name, max_diff in modified_Layers:
    print(f"  {name}: max_abs_diff = {max_diff:.6f}")
