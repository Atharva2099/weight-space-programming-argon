import torch
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

end_think_id = tokenizer.convert_tokens_to_ids("</think>")

# Identify token IDs for redundant loop starters and completion cues
loop_words = [
    "Wait", " Wait", "\nWait",
    "Alternative", " Alternative", "\nAlternative",
    "Alternatively", " Alternatively",
    "Another", " Another",
    "Hmm", " Hmm",
    "However", " However"
]
loop_ids = []
for w in loop_words:
    ids = tokenizer.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        loop_ids.append(ids[0])
loop_ids = sorted(list(set(loop_ids)))

completion_words = [
    "verified", " verified", "complete", " complete",
    "Therefore", " Therefore", "Thus", " Thus",
    "final", " final", "correct", " correct",
    "answer", " answer"
]
comp_ids = []
for w in completion_words:
    ids = tokenizer.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        comp_ids.append(ids[0])
comp_ids = sorted(list(set(comp_ids)))

print(f"Loop starter token IDs ({len(loop_ids)}): {[tokenizer.decode([i]) for i in loop_ids]}")
print(f"Completion cue token IDs ({len(comp_ids)}): {[tokenizer.decode([i]) for i in comp_ids]}")

with torch.no_grad():
    lm_w = model.lm_head.weight.data.float()
    
    # Unit direction for </think>
    w_end = lm_w[end_think_id].clone()
    u_end = w_end / w_end.norm()
    
    # Mean unit direction for loop starters and completion cues
    w_loop = lm_w[loop_ids].mean(dim=0)
    u_loop = w_loop / w_loop.norm()
    
    w_comp = lm_w[comp_ids].mean(dim=0)
    u_comp = w_comp / w_comp.norm()
    
    # Construct contrastive residual write direction: push toward </think>, pull away from loop restarts
    v_target = u_end - 0.35 * u_loop
    v_target = v_target / v_target.norm()
    
    # Detector direction for selecting completion-sensitive MLP neurons
    v_detector = 0.65 * u_end + 0.35 * u_comp
    v_detector = v_detector / v_detector.norm()

    # Modify MLP down_proj in Layers 18..23 (last 6 layers)
    layer_scales = {18: 0.08, 19: 0.10, 20: 0.14, 21: 0.18, 22: 0.22, 23: 0.26}
    top_k = 48

    for l_idx, alpha in layer_scales.items():
        down_proj = model.model.layers[l_idx].mlp.down_proj.weight.data.float()
        
        col_norms = down_proj.norm(dim=0).clamp(min=1e-6)
        cos_comp = torch.mv(down_proj.t(), v_detector) / col_norms
        cos_loop = torch.mv(down_proj.t(), u_loop) / col_norms
        neuron_scores = cos_comp - 0.25 * cos_loop
        
        top_vals, top_indices = torch.topk(neuron_scores, k=top_k)
        
        gates = torch.softmax(top_vals * 10.0, dim=0)
        mean_col_norm = col_norms[top_indices].mean()
        
        delta = torch.outer(v_target, gates) * (alpha * mean_col_norm * (top_k ** 0.5))
        down_proj[:, top_indices] += delta
        model.model.layers[l_idx].mlp.down_proj.weight.data.copy_(down_proj.to(dtype))

    # Calibrate lm_head weights for </think> vs redundant loop starters
    model.lm_head.weight.data[end_think_id] *= 1.24
    for lid in loop_ids:
        model.lm_head.weight.data[lid] *= 0.90

print("\nApplied calibrated weight patch to Layers 18-23 (MLP down_proj) and lm_head (1.24x </think>, 0.90x loop tokens).")

# Evaluate on all 4 practice problems
print("\n--- Evaluating Calibrated Model on All 4 Problems (max_new_tokens=1024) ---")
results = sandbox.forward_pass(problem_indices=[0, 1, 2, 3], max_new_tokens=1024)

success_count = 0
for res in results:
    idx = res["problem_index"]
    prob = res["prompt"]
    out = res["output"]
    has_end = "</think>" in out
    if has_end:
        success_count += 1
        parts = out.split("</think>", 1)
        think_part = parts[0]
        final_part = parts[1]
        print(f"\n[Problem {idx}] {prob[:65]}...")
        print(f"Contains </think>: True | Think len: {len(think_part)} chars | Final len: {len(final_part)} chars")
        print(f"[End of <think>]: ...{think_part[-180:].strip()}")
        print(f"[FINAL RESPONSE]:\n{final_part.strip()[:450]}")
    else:
        print(f"\n[Problem {idx}] {prob[:65]}...")
        print(f"Contains </think>: False | Total len: {len(out)} chars")
        print(f"[Tail of output]: ...{out[-250:].strip()}")

print(f"\nSummary: {success_count} / {len(results)} problems emitted </think> and produced a final answer.")
