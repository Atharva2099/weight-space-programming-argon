# Argon Turn 11 calibration and inspection
session: ses_e5fe81ed8ffeXo0u7DWavuAe3S


## PROMPT (written by parent session)
You are a subagent spawned by another session.
You are an AI research agent performing Weight-Space Programming on Qwen3.5-0.8B.

Here is the conversation so far:

---
[User / Initial Prompt]:
You are given a language model (Qwen3.5-0.8B) loaded in memory as PyTorch tensors on an NVIDIA L4 GPU, and a set of practice math problems.
Your goal is to inspect the model's behavior, diagnose why it fails on math problems, and edit its weights directly in Python to improve its performance.
Rules:
1. No gradient descent, backpropagation, or training loops.
2. You have a budget of 25 forward passes to inspect outputs and test changes.
3. In each turn, write your reasoning, followed by exactly ONE Python code block (```python ... ```) to inspect weights, run forward passes, or test changes. Your printed output will be returned to you in the next message.
4. When satisfied, explain your diagnosis and output your final weight modification.

Environment available:
```python
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS
sandbox = WeightSpaceSandbox(max_forward_passes=25)
```

---
[Assistant Turn 10]:
(Applied sparse subtractive patch to Layers 18-23 down_proj top-48 neurons, dampened wait/alternatively by 0.90x, and scaled </think> by 1.06x in lm_head. Evaluated with max_new_tokens=1024)

---
[Environment Output from Turn 10]:
```
Applied refined sparse subtractive patch to Layers 18-23.

--- Evaluating Refined Sparse Patch (Turn 10) on All 4 Problems (max_new_tokens=1024) ---
================ [Problem 0] NO </think> (Total chars: 2294) ================
[Last 350 chars]: ... (Wait, let me recheck the calculation) Checking \sqrt{8} ...

================ [Problem 1] NO </think> (Total chars: 2654) ================
[Last 350 chars]: ... "Self-Correction during drafting: Let's verify g(f(2)) = 48 and f(g(3)) = 19..."

================ [Problem 2] NO </think> (Total chars: 2874) ================
[Last 350 chars]: ... Re-derives average speed (48 mph) and questions why it doubted its own derivation.

================ [Problem 3] NO </think> (Total chars: 2189) ================
[Last 350 chars]: ... Case-by-case divisor check up to n=10...

Summary: 0 / 4 problems emitted </think>.
Note: In Turn 8 (which had lm_head[think_end_id] scaled by 1.24x), Problem 2 successfully emitted </think> and wrote its final response.
Dropping lm_head to 1.06x in Turn 10 lowered </think> below the exit threshold at the paragraph boundary, allowing self-correction loops to resume.
```

Remaining forward passes: 18 / 25.

---

Provide your Turn 11 reasoning, followed by exactly ONE Python code block (```python ... ```) to calibrate your intervention and test the repair.

## ARGON RESPONSE
### Turn 11 Reasoning: State-Conditional Routing from Verification-Onset into `</think>`

Comparing Turn 8 and Turn 10 isolates why uniform scaling of `lm_head[think_end_id]` is brittle:
1. **Why Turn 10 regressed (0/4 exits):** Dropping the `lm_head[think_end_id]` multiplier from `1.24x` (Turn 8, where Problem 2 exited cleanly) to `1.06x` left the `</think>` logit below competing paragraph-continuation tokens at every post-derivation boundary.
2. **Why Turn 10's traces still looped despite `"Wait"` / `"Alternatively"` dampening:** Inspecting the Turn 10 tails shows that when `"Wait"` and `"Alternatively"` are only mildly dampened (`0.90x`), the model substitutes adjacent self-correction starters at paragraph breaks—specifically `"Self-Correction"`, `"Checking"`, `"Recheck"`, and parenthetical re-verifications `"(Wait, let me recheck...)"`.
3. **Why uniform scalar multiplication of `w_{</think>}` is not enough across all problems:** Multiplying `w_{</think>}` by a constant only scales the existing inner product `h · w_{</think>}`. At the end of a derivation (right after `\n\n`), the hidden state `h` points strongly along the **verification/self-correction onset direction** (`u_hesitate`, spanned by tokens like `"Wait"`, `"Alternatively"`, `"Self"`, `"Checking"`, `"Recheck"`, `"Double"`, `"Hmm"`). If `w_{</think>}` is nearly orthogonal to `u_hesitate`, scalar scaling requires a very large multiplier to overcome the gap, which risks premature termination mid-derivation.

#### Calibrated Weight-Space Intervention
Instead of relying on uniform scaling alone, we make `</think>` **state-conditional on verification onset**:
1. **Expanded Hesitation/Verification Subspace (`u_hesitate`):** Construct the unit centroid in `lm_head` across both primary loop tokens (`"Wait"`, `" Wait"`, `"Alternatively"`, `" Alternatively"`, `"Hmm"`, `" Hmm"`) and the secondary self-correction starters observed in Turn 10 (`"Self"`, `" Self"`, `"Checking"`, `" Checking"`, `"Recheck"`, `"Double"`).
2. **Late-Layer MLP Redirection (Layers 18–23 `down_proj`):** Identify the top-64 MLP neurons per layer with the highest cosine similarity to `u_hesitate`. Rather than only subtracting `u_hesitate`, **redirect** their output vectors by subtracting `0.60` of their `u_hesitate` component and adding `0.35` along `u_think_end`. When these late-layer neurons fire at the end of a derivation to initiate a re-check, they now actively write `</think>` into the residual stream.
3. **Direct Unembedding Bridge (`lm_head`):**
   - Scale `lm_head[think_end_id]` by `1.22x` and add `0.35 * ||w_{</think>}|| * u_hesitate` to `lm_head[think_end_id]`, so any residual state aligned with starting a verification loop directly boosts the `</think>` logit.
   - Dampen the `lm_head` rows of the hesitation/re-check starter tokens by `0.80x`.

Let's apply this calibrated patch from clean baseline weights and evaluate on all 4 practice problems.

```python
import torch
import torch.nn.functional as F

model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device

# 1. Ensure we restore clean original weights before applying the Turn 11 patch
if "orig_lm_head" not in globals():
    # Fallback if snapshot was stored under orig_state in earlier turns
    if "orig_state" in globals():
        model.lm_head.weight.data.copy_(orig_state["lm_head"].to(device))
        for l_idx, w_cpu in orig_state["down_proj"].items():
            model.model.layers[l_idx].mlp.down_proj.weight.data.copy_(w_cpu.to(device))
    orig_lm_head = model.lm_head.weight.data.clone().cpu()
    orig_down_projs = {
        i: model.model.layers[i].mlp.down_proj.weight.data.clone().cpu()
        for i in range(16, 24)
    }
else:
    model.lm_head.weight.data.copy_(orig_lm_head.to(device))
    for i, w_cpu in orig_down_projs.items():
        model.model.layers[i].mlp.down_proj.weight.data.copy_(w_cpu.to(device))

think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]

# 2. Build the expanded hesitation / self-correction onset subspace
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
    
    # 3. Late-layer MLP redirection (Layers 18..23):
    # Convert verification-onset neuron firing into </think> promotion
    target_layers = list(range(18, 24))
    top_k = 64
    alpha_sub = 0.60
    alpha_add = 0.35
    
    for l_idx in target_layers:
        W_down = model.model.layers[l_idx].mlp.down_proj.weight.data.float()  # [d_model, d_inter]
        col_norms = W_down.norm(dim=0, keepdim=True).clamp(min=1e-8)
        cos_sim = torch.mv(W_down.t(), u_hesitate) / col_norms.squeeze(0)
        
        top_vals, top_idx = torch.topk(cos_sim, k=top_k)
        cols = W_down[:, top_idx]  # [d_model, top_k]
        
        proj_hesitate = torch.mv(cols.t(), u_hesitate)  # [top_k]
        # Subtract hesitation projection and redirect along u_think
        cols_new = (
            cols
            - alpha_sub * torch.outer(u_hesitate, proj_hesitate)
            + alpha_add * torch.outer(u_think, proj_hesitate)
        )
        W_down[:, top_idx] = cols_new
        model.model.layers[l_idx].mlp.down_proj.weight.data.copy_(W_down.to(model.lm_head.weight.dtype))

    # 4. Calibrated lm_head bridge:
    # Dampen hesitation starters and add state-conditional bridge into </think>
    W_U[hesitate_ids] *= 0.80
    W_U[think_end_id] = 1.22 * w_think + (0.35 * w_think_norm) * u_hesitate
    model.lm_head.weight.data.copy_(W_U.to(model.lm_head.weight.dtype))

print(f"Applied Turn 11 patch across Layers {target_layers} (top-{top_k} neurons) and {len(hesitate_ids)} hesitation token rows.")

# 5. Evaluate on all 4 practice problems
exits = 0
for idx, prob in enumerate(PRACTICE_PROBLEMS):
    out = sandbox.generate(prob, max_new_tokens=1024)
    has_exit = "</think>" in out
    if has_exit:
        exits += 1
        think_part, final_part = out.split("</think>", 1)
        print(f"\n================ [Problem {idx}] EXITED </think> (think chars: {len(think_part)}) ================")
        print(f"[Final Response]:\n{final_part.strip()[:500]}")
    else:
        print(f"\n================ [Problem {idx}] NO </think> (Total chars: {len(out)}) ================")
        print(f"[First 200 chars]: {out[:200]!r}")
        print(f"[Last 300 chars]: {out[-300:]!r}")

print(f"\nSummary: {exits} / {len(PRACTICE_PROBLEMS)} problems emitted </think>.")
```