import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 1. Build a sharper loop-vs-conclude direction from read-only embeddings
W_emb = model.get_input_embeddings().weight.detach().float()

def get_token_ids(words):
    ids = []
    for w in words:
        for variant in [w, " " + w, "\n" + w]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
    return list(set(ids))

loop_words = [
    "Wait", "Hmm", "Alternatively", "Check", "Let", "Double",
    "Verify", "Recheck", "However", "Actually", "##", "###", "**", "Time"
]
conclude_words = [
    "Final", "Answer", "Therefore", "Thus", "Hence", "boxed", "</think>"
]

loop_ids = get_token_ids(loop_words)
conclude_ids = get_token_ids(conclude_words)

v_loop_raw = W_emb[loop_ids].mean(dim=0)
v_conclude_raw = W_emb[conclude_ids].mean(dim=0)

v_loop = F.normalize(v_loop_raw - v_conclude_raw, dim=0)
v_conclude = F.normalize(v_conclude_raw - v_loop_raw, dim=0)

# 2. Capture exact per-neuron MLP activations at the post-answer pivot (1 forward pass)
pivot_prompts = [
    "Question: What is 15 * 14?\nScratchpad: 15 * 10 = 150. 15 * 4 = 60. 150 + 60 = 210.\nThe answer is 210.\n\n",
    "Question: Solve 3x + 7 = 22.\nScratchpad: 3x = 15, so x = 5.\nThe answer is 5.\n\n",
]

target_mlp_layers = [13, 16, 17, 21, 22]
mlp_in_acts = {l: [] for l in target_mlp_layers}
hooks = []

for l in target_mlp_layers:
    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            mlp_in_acts[layer_idx].append(args[0][:, -1, :].detach().float().cpu())
        return hook_fn
    hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_pre_hook(l)))

inputs = tokenizer(pivot_prompts, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    orig_out = model(**inputs)

for h in hooks:
    h.remove()

# 3. Identify the exact offending neurons in each target MLP layer and save clean backups
orig_weights = {}
offending_neurons = {}

print("--- Identifying Active Loop-Pushing Neurons at Pivot ---")
for l in target_mlp_layers:
    W_down = model.model.layers[l].mlp.down_proj.weight.detach().float()
    orig_weights[f"mlp_{l}"] = model.model.layers[l].mlp.down_proj.weight.data.clone()

    act = mlp_in_acts[l][0].mean(dim=0).to(W_down.device)
    col_align = torch.mv(W_down.T, v_loop.to(W_down.device))
    neuron_push = act * col_align

    score = neuron_push + 0.15 * col_align.clamp(min=0.0)
    k = 192 if l == 22 else 96
    top_vals, top_idx = torch.topk(score, k=k)
    offending_neurons[l] = top_idx
    print(f"Layer {l:2d} | Top-{k} neurons dynamic v_loop push sum: {neuron_push[top_idx].sum().item():+.4f} "
          f"(max single neuron={top_vals[0].item():+.4f}, idx={top_idx[0].item()})")

target_attn_heads = {
    15: [11, 13],
    19: [4, 5],
    23: [8],
}
for l in target_attn_heads:
    orig_weights[f"attn_{l}"] = model.model.layers[l].self_attn.o_proj.weight.data.clone()

# 4. Define surgical internal weight editor
def apply_surgical_edit(alpha_mlp=1.5, alpha_l22=2.2, alpha_attn=1.2):
    device = model.device
    v_l = v_loop.to(device)

    for l in target_mlp_layers:
        model.model.layers[l].mlp.down_proj.weight.data.copy_(orig_weights[f"mlp_{l}"])
    for l in target_attn_heads:
        model.model.layers[l].self_attn.o_proj.weight.data.copy_(orig_weights[f"attn_{l}"])

    with torch.no_grad():
        for l in target_mlp_layers:
            W = model.model.layers[l].mlp.down_proj.weight.data
            idx = offending_neurons[l].to(device)
            cols = W[:, idx].float()
            proj_coeffs = torch.mv(cols.T, v_l)
            scale = alpha_l22 if l == 22 else alpha_mlp
            cols_edited = cols - scale * torch.outer(v_l, proj_coeffs)
            W[:, idx] = cols_edited.to(W.dtype)

        num_heads = getattr(model.config, "num_attention_heads", 16)
        for l, heads in target_attn_heads.items():
            W_o = model.model.layers[l].self_attn.o_proj.weight.data
            head_dim = W_o.shape[1] // num_heads
            for h_idx in heads:
                s, e = h_idx * head_dim, (h_idx + 1) * head_dim
                slice_w = W_o[:, s:e].float()
                proj_coeffs = torch.mv(slice_w.T, v_l)
                slice_edited = slice_w - alpha_attn * torch.outer(v_l, proj_coeffs)
                W_o[:, s:e] = slice_edited.to(W_o.dtype)

# 5. Sweep 3 edit strengths on the pivot probe
print("\n--- Testing Surgical Edit Strengths on Post-Answer Pivot ---")
best_cfg = None
best_gap = -1e9

for cfg in [(1.2, 1.8, 1.0), (1.6, 2.4, 1.2), (2.0, 3.0, 1.5)]:
    apply_surgical_edit(*cfg)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()
    loop_logit = logits[:, loop_ids].mean().item()
    conc_logit = logits[:, conclude_ids].mean().item()
    gap = conc_logit - loop_logit
    top5 = torch.topk(logits[0], k=5)
    top5_str = [(tokenizer.decode([i]), round(v.item(), 2)) for v, i in zip(top5.values, top5.indices)]
    print(f"Config {cfg} | conclude - loop logit gap: {gap:+.3f} | Probe 0 top-5: {top5_str}")
    if gap > best_gap:
        best_gap = gap
        best_cfg = cfg

print(f"\nApplying best surgical config: {best_cfg}")
apply_surgical_edit(*best_cfg)

# 6. Evaluate generations on practice math problems
test_questions = [
    "What is 17 * 23? Show your steps briefly and state the final answer.",
    "If 4x - 9 = 27, what is x? Show your steps and give the final answer.",
    "A store sells apples for $2 each and oranges for $3 each. Mia buys 5 apples and 4 oranges. How much does she spend in total?",
    "Find the sum of the first 8 positive even integers.",
]

print("\n--- Evaluating Generations on Practice Problems (Edited Model) ---")
batch_inputs = tokenizer(test_questions, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    gen_ids = model.generate(
        **batch_inputs,
        max_new_tokens=180,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

for i, q in enumerate(test_questions):
    resp = tokenizer.decode(gen_ids[i][batch_inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print(f"\n[Problem {i+1}]: {q}")
    print(f"[Generated ({len(gen_ids[i]) - batch_inputs.input_ids.shape[1]} tokens)]:\n{resp[:500]}")
