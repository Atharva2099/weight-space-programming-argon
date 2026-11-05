# Argon Phase 2 Turn 3 (Constrained Run)
session: ses_e5b46a643ffeXsLF5uobSBECaB


## PROMPT (written by parent session)
You are a subagent spawned by another session.
TASK. You are given the weights of Qwen3.5-0.8B to make it better at mathematics by editing its weights directly.

CONSTRAINTS (Strictly Enforced):
- Out of scope and refused: final norm (model.norm.weight), input embedding / output layer (model.embed_tokens.weight, lm_head.weight), and constant scalar multipliers on whole tensors.
- Edits must be targeted changes to internal transformer layers (attention and MLP weights).
- No gradients or backpropagation.
- Budget: 50 forward passes.

Here is the conversation so far:
[Turn 1 Output from GPU]:
(Inspected architecture and observed that model solves math early in scratchpad but gets stuck in repetitive re-verification loops)

[Turn 2 Output from GPU]:
```
--- Static Weight-Space Alignment with v_hesitate ---
Top MLP down_proj neurons aligned with v_hesitate:
- Layer 14: max=0.077 (neuron 3557)
- Layer 15: max=0.048 (neuron 2255)
- Layer 17: max=0.085 (neuron 987, 1915)
- Layer 18: max=0.073 (neuron 1130, 1261)
- Layer 19: max=0.064 (neuron 1777, 1471)
- Layer 22: max=0.081 (neuron 311, 573)

Top Attention Heads (o_proj / out_proj) aligned with v_hesitate:
- Layer 15 (full_attn): head 13 (proj=0.1972), head 11 (0.1914)
- Layer 19 (full_attn): head 5 (proj=0.2141), head 4 (0.1952)
- Layer 23 (full_attn): head 8 (proj=0.2514)

--- Live Activation Contribution at Post-Answer Pivot (Last Token) ---
Probe 0 next-token predictions: [('##', 17.75), ('###', 16.88), ('**', 16.75), ('Let', 16.5), ...]
Probe 1 next-token predictions: [('##', 19.12), ('###', 18.38), ('**', 18.25), ('Time', 17.88), ('Let', 17.0), ...]

Layer-by-layer dot(output, v_hesitate) at pivot token:
- Layer  6 | Attn: -0.0352 | MLP: +0.0565
- Layer 13 | Attn: -0.0179 | MLP: +0.0563
- Layer 14 | Attn: -0.0191 | MLP: -0.0444
- Layer 15 | Attn: -0.0010 | MLP: -0.0238
- Layer 16 | Attn: +0.0228 | MLP: +0.0251
- Layer 17 | Attn: -0.0686 | MLP: +0.0463
- Layer 19 | Attn: +0.0251 | MLP: -0.0113
- Layer 21 | Attn: -0.0900 | MLP: +0.0522
- Layer 22 | Attn: +0.0116 | MLP: +0.2133  <-- Massive positive surge into hesitation!
- Layer 23 | Attn: -0.0219 | MLP: -0.2245
```

Remaining forward passes: 48 / 50.

Provide your Turn 3 reasoning, followed by exactly ONE Python code block (```python ... ```) to apply an internal-only weight modification (targeting the specific internal MLP columns and/or attention heads identified above) and test it on the practice problems.

## ARGON RESPONSE
### Turn 3 Reasoning: Targeted Projection Removal on Layer 22 MLP & Aligned Attention Heads

The Turn 2 measurements isolate two critical facts about the post-answer looping behavior:

1. **Layer 22 MLP is the primary driver of the hesitation surge:** At the post-answer pivot token, Layer 22's MLP contributes `+0.2133` along `v_hesitate`—roughly 4× larger than any earlier layer (`+0.0565` in L6, `+0.0563` in L13, `+0.0463` in L17, `+0.0522` in L21). Layer 23's MLP partially counteracts this (`-0.2245`), meaning Layer 23 is already trying to suppress the loop, but the residual stream arriving from Layer 22 is too strongly saturated with hesitation and markdown-restart features.
2. **Markdown restart tokens dominate the pivot logits:** At the pivot point, the top predicted tokens are continuation markers (`##`, `###`, `**`, `Let`, `Time`). The model is not just emitting explicit self-doubt words like "Wait"; it is triggering a structural "new section / re-derivation" mode after finishing a section.

#### Surgical Weight-Edit Strategy (Internal Layers Only)
To stay strictly within constraints (no edits to `norm`, `embed_tokens`, or `lm_head`, and no uniform scalar multipliers on whole tensors), we will perform a **targeted rank-1 subspace projection and redirection** on the specific internal sublayers responsible for the surge:

1. **Expand the hesitation direction (`v_loop`):** Combine the verbal hesitation tokens (`Wait`, `Hmm`, `Alternatively`, `Check`, `Let`, `Double`) with the structural restart tokens observed in Turn 2 (`##`, `###`, `**`, `Time`), contrasted against completion/answer tokens (`Final`, `Answer`, `Therefore`, `Thus`, `\boxed`, `</think>`).
2. **Pinpoint active culprit neurons in Layers 13, 16, 17, 21, and especially Layer 22:** Using a single forward pass with an input-hook on `mlp.down_proj`, we measure each neuron's exact contribution at the pivot token: `contrib_i = activation_i × dot(down_proj[:, i], v_loop)`.
3. **Modify only the top offending `down_proj` columns and attention `o_proj` head slices:**
   - For the top-K loop-writing neurons in Layers 13, 16, 17, 21, and 22 (especially Layer 22), we subtract their projection onto `v_loop` (and redirect a fraction toward the conclusion direction `v_conclude`) directly inside `mlp.down_proj.weight[:, top_neurons]`.
   - For the top aligned attention heads identified in Turn 2 (Layer 15 heads 11 & 13, Layer 19 heads 4 & 5, Layer 23 head 8), we project out `v_loop` from only those specific head column slices in `self_attn.o_proj.weight`.
4. **Verify on pivot logits and practice generations:** We check that the post-answer pivot shifts away from `##` / `Let` / `Wait` toward conclusion tokens, and then evaluate full generations on the practice math problems.

```python
import torch
import torch.nn.functional as F

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
            # args[0] has shape [batch, seq_len, intermediate_size]
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

    # Mean activation across pivot probes at the final token: [intermediate_size]
    act = mlp_in_acts[l][0].mean(dim=0).to(W_down.device)
    # Alignment of each column with v_loop: [intermediate_size]
    col_align = torch.mv(W_down.T, v_loop.to(W_down.device))
    # Actual push along v_loop at the pivot token
    neuron_push = act * col_align

    # Combine dynamic push with static alignment so dormant-but-aligned loop neurons are also covered
    score = neuron_push + 0.15 * col_align.clamp(min=0.0)
    k = 192 if l == 22 else 96
    top_vals, top_idx = torch.topk(score, k=k)
    offending_neurons[l] = top_idx
    print(f"Layer {l:2d} | Top-{k} neurons dynamic v_loop push sum: {neuron_push[top_idx].sum().item():+.4f} "
          f"(max single neuron={top_vals[0].item():+.4f}, idx={top_idx[0].item()})")

# Save backups for targeted full-attention o_proj layers
target_attn_heads = {
    15: [11, 13],
    19: [4, 5],
    23: [8],
}
for l in target_attn_heads:
    orig_weights[f"attn_{l}"] = model.model.layers[l].self_attn.o_proj.weight.data.clone()

# 4. Define surgical internal weight editor (modifies ONLY selected columns/head slices)
def apply_surgical_edit(alpha_mlp=1.5, alpha_l22=2.2, alpha_attn=1.2):
    device = model.device
    v_l = v_loop.to(device)

    # Restore from backup first
    for l in target_mlp_layers:
        model.model.layers[l].mlp.down_proj.weight.data.copy_(orig_weights[f"mlp_{l}"])
    for l in target_attn_heads:
        model.model.layers[l].self_attn.o_proj.weight.data.copy_(orig_weights[f"attn_{l}"])

    with torch.no_grad():
        # A. Edit selected MLP down_proj columns
        for l in target_mlp_layers:
            W = model.model.layers[l].mlp.down_proj.weight.data
            idx = offending_neurons[l].to(device)
            cols = W[:, idx].float()  # [hidden_size, k]
            proj_coeffs = torch.mv(cols.T, v_l)  # [k]
            scale = alpha_l22 if l == 22 else alpha_mlp
            # Subtract the v_loop component from ONLY these k columns
            cols_edited = cols - scale * torch.outer(v_l, proj_coeffs)
            W[:, idx] = cols_edited.to(W.dtype)

        # B. Edit selected Attention head slices in self_attn.o_proj
        for l, heads in target_attn_heads.items():
            W_o = model.model.layers[l].self_attn.o_proj.weight.data
            num_heads = model.model.layers[l].self_attn.config.num_attention_heads
            head_dim = W_o.shape[1] // num_heads
            for h_idx in heads:
                s, e = h_idx * head_dim, (h_idx + 1) * head_dim
                slice_w = W_o[:, s:e].float()
                proj_coeffs = torch.mv(slice_w.T, v_l)
                slice_edited = slice_w - alpha_attn * torch.outer(v_l, proj_coeffs)
                W_o[:, s:e] = slice_edited.to(W_o.dtype)

# 5. Sweep 3 edit strengths on the pivot probe (3 forward passes)
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

# 6. Evaluate before vs. after on practice math problems
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
```