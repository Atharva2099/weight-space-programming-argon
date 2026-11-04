import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("--- Inspecting Token IDs for Loop vs Conclusion ---")

loop_words = ["Wait", " Wait", "Alternatively", " Alternatively", "Let's", " Let's", "But", " But", "Hmm", "Check", " Double"]
conclude_words = ["Therefore", " Therefore", "Thus", " Thus", "Final", " Final", "Answer", " Answer", "\\boxed", "</think>"]

def get_single_token_ids(word_list):
    ids = []
    for w in word_list:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) >= 1:
            ids.append((w, toks[0], tokenizer.decode([toks[0]])))
    seen = set()
    dedup = []
    for w, tid, dec in ids:
        if tid not in seen:
            seen.add(tid)
            dedup.append((w, tid, dec))
    return dedup

loop_toks = get_single_token_ids(loop_words)
conclude_toks = get_single_token_ids(conclude_words)
if tokenizer.eos_token_id is not None:
    conclude_toks.append(("<eos>", tokenizer.eos_token_id, tokenizer.decode([tokenizer.eos_token_id])))

print("Loop tokens:", [(dec, tid) for _, tid, dec in loop_toks])
print("Conclude tokens:", [(dec, tid) for _, tid, dec in conclude_toks])

# Construct normalized contrastive direction in residual space
W_U = model.lm_head.weight.detach().float()
loop_ids = [tid for _, tid, _ in loop_toks]
conclude_ids = [tid for _, tid, _ in conclude_toks]

v_loop = F.normalize(W_U[loop_ids].mean(dim=0), dim=0)
v_conclude = F.normalize(W_U[conclude_ids].mean(dim=0), dim=0)
v_hesitate = F.normalize(v_loop - v_conclude, dim=0)

print("\n--- Static Weight-Space Alignment with v_hesitate ---")
layer_mlp_scores = []
layer_attn_scores = []

for i, layer in enumerate(model.model.layers):
    W_down = layer.mlp.down_proj.weight.detach().float()
    mlp_dots = torch.mv(W_down.T, v_hesitate)
    top_pos_val, top_pos_idx = torch.topk(mlp_dots, 5)
    layer_mlp_scores.append((i, mlp_dots.abs().max().item(), mlp_dots.norm().item(), top_pos_idx.tolist(), top_pos_val.tolist()))

    if hasattr(layer, "self_attn"):
        W_o = layer.self_attn.o_proj.weight.detach().float()
        kind = "full_attn"
    else:
        W_o = layer.linear_attn.out_proj.weight.detach().float()
        kind = "lin_attn"
    
    n_heads = 16
    head_dim = W_o.shape[1] // n_heads
    head_norms = []
    for h in range(n_heads):
        W_h = W_o[:, h * head_dim : (h + 1) * head_dim]
        proj_norm = torch.mv(W_h.T, v_hesitate).norm().item()
        head_norms.append((h, round(proj_norm, 4)))
    head_norms.sort(key=lambda x: x[1], reverse=True)
    layer_attn_scores.append((i, kind, head_norms[:3]))

for i, max_v, norm_v, idxs, vals in layer_mlp_scores:
    print(f"Layer {i:2d} MLP | norm={norm_v:.3f} max={max_v:.3f} | top neurons={idxs[:3]} ({[round(x,3) for x in vals[:3]]})")

print("\nTop Attention Heads aligned with v_hesitate:")
for i, kind, top_heads in layer_attn_scores:
    print(f"Layer {i:2d} ({kind:9s}) | top heads (head_idx, proj_norm): {top_heads}")

# Run 1 Forward Pass on a batch of post-solution pivot traces
probe_texts = [
    "g(3) = 2(3) + 2 = 8. f(8) = 2(8) + 3 = 19. f(2) = 2(2) + 3 = 7. g(7) = 7^2 - 1 = 48. 19 - 48 = -29.\n\n",
    "Time there = 120 / 40 = 3 hours. Time back = 120 / 60 = 2 hours. Total distance = 240 miles. Total time = 5 hours. Average speed = 240 / 5 = 48 mph.\n\n",
]

inputs = tokenizer(probe_texts, return_tensors="pt", padding=True).to(model.device)
attn_contribs = {}
mlp_contribs = {}

hooks = []
for i, layer in enumerate(model.model.layers):
    def make_attn_hook(layer_idx):
        def hook(module, inp, out):
            attn_contribs[layer_idx] = out.detach().float()
        return hook

    def make_mlp_hook(layer_idx):
        def hook(module, inp, out):
            mlp_contribs[layer_idx] = out.detach().float()
        return hook

    if hasattr(layer, "self_attn"):
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_attn_hook(i)))
    else:
        hooks.append(layer.linear_attn.out_proj.register_forward_hook(make_attn_hook(i)))
    hooks.append(layer.mlp.down_proj.register_forward_hook(make_mlp_hook(i)))

with torch.no_grad():
    outputs = model(**inputs)

for h in hooks:
    h.remove()

v_hes_dev = v_hesitate.to(outputs.logits.device)
print("\n--- Live Activation Contribution at Post-Answer Pivot (Last Token) ---")
for b in range(len(probe_texts)):
    seq_len = inputs.attention_mask[b].sum().item() - 1
    top_logits, top_ids = torch.topk(outputs.logits[b, seq_len], 8)
    print(f"\nProbe {b} next-token predictions:", [(tokenizer.decode([tid.item()]), round(val.item(), 2)) for val, tid in zip(top_logits, top_ids)])

print("\nLayer-by-layer dot(output, v_hesitate) averaged across probes at pivot token:")
for i in range(len(model.model.layers)):
    a_scores = []
    m_scores = []
    for b in range(len(probe_texts)):
        seq_len = inputs.attention_mask[b].sum().item() - 1
        a_out = attn_contribs[i][b, seq_len].to(v_hes_dev.device)
        m_out = mlp_contribs[i][b, seq_len].to(v_hes_dev.device)
        a_scores.append(torch.dot(a_out, v_hes_dev).item())
        m_scores.append(torch.dot(m_out, v_hes_dev).item())
    print(f"Layer {i:2d} | Attn->Hesitate: {sum(a_scores)/len(a_scores):+.4f} | MLP->Hesitate: {sum(m_scores)/len(m_scores):+.4f}")
