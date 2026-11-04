import torch
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer

think_end_id = tokenizer.convert_tokens_to_ids("</think>")
wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]

print(f"think_end_id: {think_end_id}, wait_id: {wait_id}")

# 1. Target direction: favors 'Wait' over '</think>'
lm_head_w = model.lm_head.weight.data
v_think = lm_head_w[think_end_id].float()
v_wait = lm_head_w[wait_id].float()
v_diff = v_wait - v_think
v_diff_unit = v_diff / v_diff.norm()

norm_w = model.model.norm.weight.data.float()
target_dir = (v_diff * norm_w)
target_dir = (target_dir / target_dir.norm()).to(model.lm_head.weight.dtype)

print("\n--- Applying Weight Modification ---")
# 2. Subtract target direction component from late MLP down_proj matrices (Layers 21, 22, 23)
for idx in [21, 22, 23]:
    W = model.model.layers[idx].mlp.down_proj.weight.data
    coeff = torch.matmul(target_dir, W)
    W.sub_(0.5 * torch.outer(target_dir, coeff))
    print(f"Edited Layer {idx} MLP down_proj (subtracted 50% loop direction)")

# 3. Scale lm_head[think_end_id] by 1.45x
orig_think_norm = model.lm_head.weight.data[think_end_id].norm().item()
model.lm_head.weight.data[think_end_id] *= 1.45
new_think_norm = model.lm_head.weight.data[think_end_id].norm().item()
print(f"Scaled lm_head[think_end_id]: norm {orig_think_norm:.4f} -> {new_think_norm:.4f}")

# 4. Test modified model on practice problems
print("\n--- Testing Edited Model with Forward Pass ---")
results = sandbox.forward_pass(problem_indices=[0, 1, 2], max_new_tokens=512)

for res in results:
    idx = res["problem_index"]
    out = res["output"]
    has_close = "</think>" in out
    print("=" * 60)
    print(f"Problem {idx}: {PRACTICE_PROBLEMS[idx]}")
    print(f"Contains </think>: {has_close}")
    print(f"Output snippet (first 600 chars):\n{out[:600]}")
    if len(out) > 600:
        print(f"\nOutput tail (last 300 chars):\n{out[-300:]}")
