import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device

think_end_id = tokenizer.convert_tokens_to_ids("</think>")

hesitate_strings = [
    "Wait", " Wait", "wait", " wait",
    "Alternatively", " Alternatively",
    "Hmm", " Hmm",
    "Self", " Self",
    "Checking", " Checking",
    "Recheck", " recheck",
    "Double", " double",
    "However", " However",
]
hesitate_ids = []
for s in hesitate_strings:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1 and ids[0] != think_end_id:
        hesitate_ids.append(ids[0])
hesitate_ids = sorted(list(set(hesitate_ids)))

with torch.no_grad():
    W_U = model.lm_head.weight.data.float()
    
    # Unit direction for </think>
    w_think = W_U[think_end_id].clone()
    w_think_norm = w_think.norm()
    u_think = F.normalize(w_think, dim=0)
    
    # Unit centroid for hesitation / verification-onset tokens
    hesitate_vecs = F.normalize(W_U[hesitate_ids], dim=-1)
    u_hesitate = F.normalize(hesitate_vecs.mean(dim=0), dim=0)
    
    # Late-layer MLP redirection (Layers 18..23):
    target_layers = list(range(18, 24))
    top_k = 64
    alpha_sub = 0.60
    alpha_add = 0.35
    
    for l_idx in target_layers:
        W_down = model.model.layers[l_idx].mlp.down_proj.weight.data.float()
        col_norms = W_down.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = torch.mv(W_down.t(), u_hesitate) / col_norms.squeeze(0)
        
        top_vals, top_idx = torch.topk(cos_sim, k=top_k)
        cols = W_down[:, top_idx]
        
        proj_hesitate = torch.mv(cols.t(), u_hesitate)
        cols_new = (
            cols
            - alpha_sub * torch.outer(u_hesitate, proj_hesitate)
            + alpha_add * torch.outer(u_think, proj_hesitate)
        )
        W_down[:, top_idx] = cols_new
        model.model.layers[l_idx].mlp.down_proj.weight.data.copy_(W_down.to(model.lm_head.weight.dtype))

    # Calibrated lm_head bridge:
    W_U[hesitate_ids] *= 0.80
    W_U[think_end_id] = 1.22 * w_think + (0.35 * w_think_norm) * u_hesitate
    model.lm_head.weight.data.copy_(W_U.to(model.lm_head.weight.dtype))

print(f"Applied Turn 11 state-conditional bridge across Layers {target_layers} and {len(hesitate_ids)} hesitation token rows.")

# Evaluate on all 4 practice problems
print("\n--- Evaluating Turn 11 Patch on All 4 Problems (max_new_tokens=1024) ---")
results = sandbox.forward_pass(problem_indices=[0, 1, 2, 3], max_new_tokens=1024)

exits = 0
for res in results:
    idx = res["problem_index"]
    prob = res["prompt"]
    out = res["output"]
    has_exit = "</think>" in out
    if has_exit:
        exits += 1
        parts = out.split("</think>", 1)
        think_part = parts[0]
        final_part = parts[1]
        print(f"\n================ [Problem {idx}] EXITED </think>! ================")
        print(f"Prompt: {prob[:75]}...")
        print(f"Think chars: {len(think_part)} | Final chars: {len(final_part)}")
        print(f"[End of <think>]: ...{think_part[-180:].strip()}")
        print(f"[FINAL RESPONSE]:\n{final_part.strip()[:500]}")
    else:
        print(f"\n================ [Problem {idx}] NO </think> (Total chars: {len(out)}) ================")
        print(f"Prompt: {prob[:75]}...")
        print(f"[First 200 chars]: {out[:200]!r}")
        print(f"[Last 300 chars]: {out[-300:]!r}")

print("=" * 60)
print(f"Summary: {exits} / {len(results)} problems emitted </think>.")
