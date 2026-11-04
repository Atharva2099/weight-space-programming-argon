import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer

# 1. Inspect relevant token IDs
target_strs = ["</think>", "<think>", "Wait", " Wait", "\nWait", "Let", " Let", "\n\n", "Therefore", "Final"]
token_info = {}
for s in target_strs:
    ids = tokenizer.encode(s, add_special_tokens=False)
    token_info[s] = (ids, [tokenizer.decode([i]) for i in ids])
    print(f"String {repr(s):15s} -> IDs: {ids} ({token_info[s][1]})")

think_end_id = tokenizer.convert_tokens_to_ids("</think>")
print(f"\nconvert_tokens_to_ids('</think>'): {think_end_id}")

# 2. Weight-space inspection: lm_head and MLP down_proj alignment with </think>
lm_head = model.lm_head.weight.detach().float()  # [vocab_size, hidden_size]
v_end = lm_head[think_end_id]  # [hidden_size]
v_end_norm = F.normalize(v_end, dim=0)

wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]
v_wait = lm_head[wait_id]
v_wait_norm = F.normalize(v_wait, dim=0)

print(f"</think> vector norm: {v_end.norm().item():.4f}, Wait vector norm: {v_wait.norm().item():.4f}")
print(f"Cosine(</think>, Wait): {F.cosine_similarity(v_end_norm, v_wait_norm, dim=0).item():.4f}")

# Check top tokens closest to </think> in lm_head
cos_all = torch.mv(F.normalize(lm_head, dim=1), v_end_norm)
top_vals, top_idxs = torch.topk(cos_all, 15)
print("\nTop 15 tokens aligned with </think> in lm_head:")
for val, idx in zip(top_vals, top_idxs):
    print(f"  ID {idx.item():6d} ({repr(tokenizer.decode([idx.item()])):18s}): cos={val.item():.4f}, norm={lm_head[idx].norm().item():.4f}")

# Check MLP down_proj columns across all 24 layers for alignment with </think>
print("\nLayer-wise max/min MLP down_proj alignment with </think>:")
for l_idx, layer in enumerate(model.model.layers):
    down_w = layer.mlp.down_proj.weight.detach().float()  # [hidden_size, intermediate_size]
    dots = torch.mv(down_w.T, v_end_norm)
    max_val, max_n = dots.max(dim=0)
    min_val, min_n = dots.min(dim=0)
    print(f"  Layer {l_idx:2d}: max_dot={max_val.item():+.4f} (n={max_n.item():4d}), min_dot={min_val.item():+.4f} (n={min_n.item():4d})")
