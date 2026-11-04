import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer

think_end_id = tokenizer.convert_tokens_to_ids("</think>")

# 1. Apply Turn 6 calibrated weight edit
loop_words = ["Wait", " Wait", "Hmm", "Alternatively", "Double", "Let's", "Check"]
loop_ids = []
for w in loop_words:
    ids = tokenizer.encode(w, add_special_tokens=False)
    if ids:
        loop_ids.append(ids[0])
loop_ids = list(dict.fromkeys(loop_ids))

with torch.no_grad():
    W_u = model.lm_head.weight
    v_end = F.normalize(W_u[think_end_id].float(), dim=0)

    v_loops = F.normalize(W_u[loop_ids].float(), dim=1).mean(dim=0)
    v_loops = F.normalize(v_loops, dim=0)

    v_loop_orth = v_loops - torch.dot(v_loops, v_end) * v_end
    v_loop_orth = F.normalize(v_loop_orth, dim=0).to(W_u.dtype)
    v_end_dt = v_end.to(W_u.dtype)

    for layer_idx in [20, 21, 22, 23]:
        W = model.model.layers[layer_idx].mlp.down_proj.weight
        proj_loop = torch.mv(W.t(), v_loop_orth)
        W.add_(torch.outer(v_loop_orth, proj_loop), alpha=-0.65)
        W.add_(torch.outer(v_end_dt, proj_loop), alpha=0.25)

    W_u[think_end_id].mul_(1.15)

print("Applied refined weight patch to Layers 20-23 and lm_head.")

# 2. Evaluate all 4 problems with max_new_tokens=1024
print("\n--- Evaluating Modified Model on All Problems (max_new_tokens=1024) ---")
results = sandbox.forward_pass(problem_indices=[0, 1, 2, 3], max_new_tokens=1024)

completed_think_count = 0
for res in results:
    i = res["problem_index"]
    prob = res["prompt"]
    out = res["output"]
    has_think_end = "</think>" in out
    if has_think_end:
        completed_think_count += 1
        parts = out.split("</think>", 1)
        think_part = parts[0]
        final_part = parts[1]
    else:
        think_part, final_part = out, ""

    print("=" * 60)
    print(f"Problem {i}: {prob}")
    print(f"Contains </think>: {has_think_end} | Length: {len(out)} chars")
    if has_think_end:
        print(f"\n[End of <think> snippet]: ...{think_part[-200:].strip()}")
        print(f"\n[FINAL RESPONSE AFTER </think>]:\n{final_part.strip()}")
    else:
        print(f"\n[Tail of unclosed output]:\n...{out[-350:].strip()}")

print("=" * 60)
print(f"Summary: {completed_think_count}/{len(results)} problems successfully emitted </think> within 1024 tokens!")
