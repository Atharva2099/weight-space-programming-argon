import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device

think_end_id = tokenizer.convert_tokens_to_ids("</think>")

# 1. Token 1 Logit Check on Problem 0 to ensure no premature exit
messages = [{"role": "user", "content": PRACTICE_PROBLEMS[0]}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

with torch.no_grad():
    out = model(**inputs)
    first_step_logits = out.logits[0, -1, :].float()
    top5_vals, top5_ids = torch.topk(first_step_logits, 5)
    think_end_logit_t1 = first_step_logits[think_end_id].item()

print(f"[Token 1 Logit Check]")
print(f"Top-5 starter tokens: {[(tokenizer.decode([i]), round(v.item(), 2)) for v, i in zip(top5_vals, top5_ids)]}")
print(f"</think> logit at token 1: {think_end_logit_t1:.2f} (gap to top1: {top5_vals[0].item() - think_end_logit_t1:.2f})")

# 2. Build multi-token loop direction and orthogonalize against </think>
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

    # Mean normalized direction of loop-trigger tokens
    v_loops = F.normalize(W_u[loop_ids].float(), dim=1).mean(dim=0)
    v_loops = F.normalize(v_loops, dim=0)

    # Orthogonalize v_loops against v_end
    v_loop_orth = v_loops - torch.dot(v_loops, v_end) * v_end
    v_loop_orth = F.normalize(v_loop_orth, dim=0).to(W_u.dtype)
    v_end_dt = v_end.to(W_u.dtype)

    # In Layers 20-23, damp the orthogonal loop direction by 65%,
    # and redirect 25% of that loop energy directly into v_end
    print("\n--- Applying Refined Weight Intervention (Layers 20-23) ---")
    for layer_idx in [20, 21, 22, 23]:
        W = model.model.layers[layer_idx].mlp.down_proj.weight
        proj_loop = torch.mv(W.t(), v_loop_orth)
        W.add_(torch.outer(v_loop_orth, proj_loop), alpha=-0.65)
        W.add_(torch.outer(v_end_dt, proj_loop), alpha=0.25)
        print(f"Layer {layer_idx}: Damped loop direction (65%) and redirected 25% to </think>")

    # Safe 1.15x boost to lm_head[think_end_id]
    W_u[think_end_id].mul_(1.15)
    print(f"Scaled lm_head[think_end_id] by 1.15x")

# 3. Test on Practice Problems 0, 1, 2
print("\n--- Testing Refined Model with Forward Pass ---")
results = sandbox.forward_pass(problem_indices=[0, 1, 2], max_new_tokens=512)

for res in results:
    idx = res["problem_index"]
    text = res["output"]
    has_end = "</think>" in text
    print("=" * 60)
    print(f"Problem {idx}: {PRACTICE_PROBLEMS[idx]}")
    print(f"Contains </think>: {has_end}")
    if has_end:
        parts = text.split("</think>", 1)
        print(f"[Thinking trace length: {len(parts[0])} chars]")
        print(f"[Thinking snippet]:\n{parts[0][:300]}...")
        print(f"\n[Final Answer segment]:\n{parts[1][:400]}")
    else:
        print(f"[Output tail (last 350 chars)]:\n{text[-350:]}")
