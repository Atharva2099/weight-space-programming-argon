# Argon Turn 10 diagnosis of loop regression
session: ses_e5ff17d40ffe7hWj03dOGrr1jg


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
[Assistant Turn 8 & 9]:
(In Turn 8, targeted subtraction on Layers 18-23 top-48 neurons allowed Problem 2 to cleanly emit </think> and output its final answer. In Turn 9, broadened additive steering across 8 layers 16-23 was tested with max_new_tokens=1536)

---
[Environment Output from Turn 9]:
```
Applied finalized weight-space repair to Layers 16-23 (MLP down_proj) and lm_head.

--- Evaluating Finalized Model on All 4 Problems (max_new_tokens=1536) ---
================ [Problem 0] ================
Contains </think>: False | Total chars: 4647 (~1536 tokens)
[Tail of output]: ... Ends in a repeating run of "Wait, Wait, Wait, Wait, ..."

================ [Problem 1] ================
Contains </think>: False | Total chars: 4824 (~1536 tokens)
[Tail of output]: ... Repeats "48 is correct, but the previous thought said 48. Wait... Wait... Wait..."

================ [Problem 2] ================
Contains </think>: False | Total chars: 4846 (~1536 tokens)
[Tail of output]: ... "Wait, Wait, Wait, Wait. I am confusing myself again." Then restarts the entire problem setup from scratch!

================ [Problem 3] ================
Contains </think>: False | Total chars: 3655 (~1200 tokens)
[Tail of output]: ... Cycles through algebra with "Wait, Wait, ..."

Summary: 0 / 4 problems emitted </think>. Broadening the intervention to an additive boost across 8 layers (Layers 16-23) caused a severe regression back into the classic degenerative attractor loop ("Wait, Wait, Wait...").
In contrast, Turn 8's targeted subtraction on Layers 18-23 had successfully broken Problem 2 out of the loop and brought Problem 1 to the final answer threshold.
```

Remaining forward passes: 19 / 25.

---

Provide your Turn 10 reasoning, followed by exactly ONE Python code block (```python ... ```) to analyze this regression, refine your targeted weight patch, and evaluate the result.

## ARGON RESPONSE
### Turn 10 Reasoning: Why Broad Additive Steering Regressed vs. Sparse Subtractive Repair

The regression in Turn 9 reveals an important structural property of how MLP `down_proj` columns interact with the residual stream in Qwen3.5-0.8B:

1. **Additive steering across all columns injects out-of-distribution residual energy**: When we added a dense rank-1 component `(v_exit @ g_up.T)` across 8 layers (`16–23`), every firing neuron in the intermediate activation `h_t` contributed to pushing the residual stream along `v_exit` at *every* token step—even in the middle of algebraic steps. Worse, modifying mid-upper layers (`16–17`) perturbs the representation before final-layer RMSNorm (`model.norm`), causing the model's attention and MLP heads in subsequent layers (`18–23`) to lose numeric coherence ("I am confusing myself again") and fall into the degenerative `"Wait, Wait, Wait..."` repetition attractor.
2. **Why Turn 8's sparse subtractive intervention worked**: In Turn 8, restricting the intervention strictly to the **late layers (`18–23`)** and **only the top-K (`K = 48`) conflict-aligned neurons** (`down_proj` columns with the highest positive alignment to `u_wait - u_exit`, gated by `up_proj` alignment to hesitation context) left general mathematical reasoning untouched while selectively removing the "keep-thinking / self-doubt" kick whenever those specific hesitation-detector neurons fired.
3. **Why Turn 8 only exited on Problem 2 (`1/4`)**: In Turn 8, we only scaled the subtractive suppression on `down_proj` columns (`alpha = 0.28`) without directly addressing two bottlenecks at the decision boundary:
   - **Self-reinforcing attention/MLP loop on `"Wait"` tokens**: When the model finishes computing an answer (e.g., `"48"` in Problem 1) and emits a newline `\n\n`, the `"Wait"` token competes directly with `"</think>"` in the final unembedding (`lm_head` / `embed_tokens`).
   - **Norm attenuation by final RMSNorm**: Any change made in layers `18–23` passes through `model.norm` before multiplying `lm_head`. Meanwhile, the final layer (`Layer 23`) and the unembedding (`lm_head`, or equivalently the final RMSNorm weight / late-layer output projections) directly govern the logit gap `logit("</think>") - logit("Wait")`.

#### Hypothesis & Refinement Plan
First, let's restore the clean base weights and inspect the exact token IDs and unembedding rows (`lm_head` / `embed_tokens`) for `"</think>"`, `"Wait"`, `"Wait,"`, `"Alternatively"`, and `"Let"`.
Second, let's reinstate the **Turn 8 sparse subtractive patch on Layers 18–23 (top-48 hesitation neurons)** that cleanly solved Problem 2, and combine it with a **orthogonalized late-layer exit-channel adjustment** specifically in Layers 21–23 (where final token logits are crystallized) and check how the logit margin `logit("</think>") - logit("Wait")` evolves on the actual generated states of Problem 1 and Problem 2.

To stay disciplined with our remaining 19 forward passes, we will use **4 forward passes** in this turn (`max_new_tokens=1024`) to test the refined sparse subtractive patch on all 4 problems and inspect their exact termination behavior and answers.

```python
import torch
import torch.nn.functional as F

# 1. Reset to clean original weights first
sandbox.reset_weights()
model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device

# 2. Inspect exact token IDs for exit vs hesitation starters
tokens_of_interest = ["</think>", "Wait", " Wait", "\nWait", "Alternatively", "However", "Let's", "**Final"]
tok_map = {}
for t in tokens_of_interest:
    ids = tokenizer.encode(t, add_special_tokens=False)
    tok_map[t] = ids
print("Token ID mapping:", tok_map)

id_exit = tokenizer.encode("</think>", add_special_tokens=False)[0]
id_wait1 = tokenizer.encode("Wait", add_special_tokens=False)[0]
id_wait2 = tokenizer.encode(" Wait", add_special_tokens=False)[0]
id_alt = tokenizer.encode("Alternatively", add_special_tokens=False)[0]
id_nl = tokenizer.encode("\n\n", add_special_tokens=False)[-1]

# 3. Extract normalized unembedding / embedding directions
W_U = model.lm_head.weight.detach().float()  # [vocab, d_model]
W_E = model.model.embed_tokens.weight.detach().float()

u_exit = F.normalize(W_U[id_exit], dim=0)
u_wait = F.normalize(0.6 * W_U[id_wait1] + 0.4 * W_U[id_wait2] + 0.2 * W_U[id_alt], dim=0)

# Orthogonalize u_wait and u_exit so we cleanly separate "suppress Wait" from "promote </think>"
u_conflict = F.normalize(u_wait - u_exit, dim=0)
# Component of u_exit orthogonal to u_wait
u_exit_orth = F.normalize(u_exit - torch.dot(u_exit, u_wait) * u_wait, dim=0)

# Context direction when model is about to loop (after newline / hesitation)
e_context = F.normalize(0.5 * W_E[id_wait1] + 0.3 * W_E[id_wait2] + 0.2 * W_E[id_nl], dim=0)

# 4. Apply Turn 8's proven sparse subtractive patch on Layers 18..23,
#    enhanced with stronger late-layer (20..23) suppression of the Wait direction
target_layers = [18, 19, 20, 21, 22, 23]
K_neurons = 48

with torch.no_grad():
    for l_idx in target_layers:
        mlp = model.model.layers[l_idx].mlp
        W_down = mlp.down_proj.weight.data.float()  # [d_model, d_inter]
        W_up = mlp.up_proj.weight.data.float()      # [d_inter, d_model]
        W_gate = mlp.gate_proj.weight.data.float()  # [d_inter, d_model]

        # Score each intermediate neuron j by:
        # (a) how strongly its output column pushes toward u_conflict (Wait - </think>)
        out_align = torch.mv(W_down.T, u_conflict)  # [d_inter]
        # (b) how strongly its gate/up rows activate on hesitation/newline context
        in_align = 0.5 * torch.mv(W_up, e_context) + 0.5 * torch.mv(W_gate, e_context)
        in_gate = torch.clamp(in_align, min=0.0)

        neuron_score = out_align * (0.5 + in_gate)
        top_vals, top_idx = torch.topk(neuron_score, k=K_neurons)

        # Layer-dependent scaling: gentle in 18-19, firmer in 20-23
        alpha_layer = 0.28 if l_idx < 20 else 0.42

        for rank, j in enumerate(top_idx):
            col = W_down[:, j]
            col_norm = torch.norm(col)
            # Project out the conflict direction and tilt toward u_exit
            proj_conflict = torch.dot(col, u_conflict)
            if proj_conflict > 0:
                # Remove the Wait-pushing component and replace a fraction with u_exit_orth
                col_new = col - (1.0 + alpha_layer) * proj_conflict * u_conflict + (0.35 * alpha_layer) * col_norm * u_exit_orth
                W_down[:, j] = col_new

        mlp.down_proj.weight.data.copy_(W_down.to(mlp.down_proj.weight.dtype))

    # 5. Direct unembedding / logit-boundary calibration on the conflict tokens:
    # Slightly attenuate the raw unembedding norm of "Wait" / " Wait" / "Alternatively"
    # and slightly sharpen "</think>" along the conflict axis so that when the model
    # finishes a verification block at \n\n, "</think>" wins the argmax/sampling race.
    w_u_dtype = model.lm_head.weight.dtype
    W_U_mod = model.lm_head.weight.data.float()
    for wid, scale in [(id_wait1, 0.90), (id_wait2, 0.90), (id_alt, 0.92)]:
        W_U_mod[wid] = W_U_mod[wid] * scale
    W_U_mod[id_exit] = W_U_mod[id_exit] * 1.06
    model.lm_head.weight.data.copy_(W_U_mod.to(w_u_dtype))
    # Note: if tie_word_embeddings is True, lm_head and embed_tokens share storage;
    # verify whether they are tied:
    print("tie_word_embeddings:", getattr(model.config, "tie_word_embeddings", None),
          "Same data_ptr:", model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())

# 6. Evaluate on all 4 practice problems (4 forward passes, max_new_tokens=1024)
print("\n--- Evaluating Refined Sparse Patch (Turn 10) on All 4 Problems (max_new_tokens=1024) ---")
exited_count = 0
for i, prob in enumerate(PRACTICE_PROBLEMS):
    out = sandbox.generate(prob, max_new_tokens=1024)
    has_exit = "</think>" in out
    if has_exit:
        exited_count += 1
        think_part, post_think = out.split("</think>", 1)
        print(f"\n================ [Problem {i}] EXITED </think>! ================")
        print(f"Think chars: {len(think_part)} | Post-think answer:\n{post_think.strip()[:500]}")
    else:
        print(f"\n================ [Problem {i}] NO </think> (Total chars: {len(out)}) ================")
        print(f"[Last 350 chars]:\n{out[-350:]}")

print(f"\nSummary: {exited_count} / {len(PRACTICE_PROBLEMS)} problems emitted </think>.")
```