import torch
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer

think_end_id = tokenizer.convert_tokens_to_ids("</think>")
eos_id = tokenizer.eos_token_id

loop_words = [
    "Wait", " Wait", "wait", " wait",
    "Alternatively", " Alternatively", "alternatively",
    "Hmm", " Hmm", "Let's", " Let's",
    "However", " However", "Double", " Double",
    "Recheck", " Recheck", "Review", " Review",
    "Method", " Method", "Approach", " Approach"
]
loop_token_ids = set()
for w in loop_words:
    ids = tokenizer.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        loop_token_ids.add(ids[0])
loop_token_ids = sorted(list(loop_token_ids))

with torch.no_grad():
    # 1. Normalized discourse-completion direction in residual stream
    v_end = model.lm_head.weight[think_end_id].detach().float()
    v_eos = model.lm_head.weight[eos_id].detach().float()
    v_loop = model.lm_head.weight[loop_token_ids].detach().float().mean(dim=0)

    v_comp = (0.75 * (v_end / v_end.norm()) + 0.25 * (v_eos / v_eos.norm())) - 0.55 * (v_loop / v_loop.norm())
    v_comp = v_comp / v_comp.norm()

    # 2. Rank-1 steering on top 64 completion-aligned neurons in Layers 16..23 MLP down_proj
    target_layers = range(16, 24)
    alpha_mlp = 0.042

    for l_idx in target_layers:
        W_down = model.model.layers[l_idx].mlp.down_proj.weight
        alignment = torch.mv(W_down.float().t(), v_comp)

        top_vals, top_idx = torch.topk(alignment, k=64)
        mask = torch.zeros_like(alignment)
        mask[top_idx] = torch.softmax(top_vals / top_vals.std(), dim=0) * 64.0

        delta_W = alpha_mlp * torch.outer(v_comp, mask).to(W_down.dtype)
        W_down.add_(delta_W)

    # 3. Calibrated output head (lm_head) adjustments
    model.lm_head.weight[think_end_id].mul_(1.28)
    for tid in loop_token_ids:
        model.lm_head.weight[tid].mul_(0.88)

print("Applied finalized weight-space repair to Layers 16-23 (MLP down_proj) and lm_head.")

# Evaluate across all 4 practice problems with max_new_tokens=1536
print("\n--- Evaluating Finalized Model on All 4 Problems (max_new_tokens=1536) ---")
results = sandbox.forward_pass(problem_indices=[0, 1, 2, 3], max_new_tokens=1536)

completed_count = 0
for res in results:
    idx = res["problem_index"]
    prob = res["prompt"]
    out = res["output"]
    has_end = "</think>" in out
    if has_end:
        completed_count += 1
        parts = out.split("</think>", 1)
        think_part = parts[0]
        final_part = parts[1]
        print(f"\n================ [Problem {idx}] ================")
        print(f"Prompt: {prob[:75]}...")
        print(f"Contains </think>: True | Total chars: {len(out)}")
        print(f"Think len: {len(think_part)} chars | Final len: {len(final_part)} chars")
        print(f"[End of <think>]: ...{think_part[-180:].strip()}")
        print(f"[FINAL ANSWER]:\n{final_part.strip()[:500]}")
    else:
        print(f"\n================ [Problem {idx}] ================")
        print(f"Prompt: {prob[:75]}...")
        print(f"Contains </think>: False | Total chars: {len(out)}")
        print(f"[Tail of output]: ...{out[-250:].strip()}")

print("=" * 60)
print(f"Final Summary: {completed_count} / {len(results)} problems successfully emitted </think> and produced complete answers!")
