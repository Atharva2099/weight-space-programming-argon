# Weight-Space Programming with Gemini 4 Argon: Unguided Qwen3.5-0.8B Loop Repair

Independent replication and mechanistic analysis of Experiment 8 from MIT CSAIL's *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"* (Takehi et al., Nov 2026), evaluated with **Google Gemini 4 Argon** under a strictly blind, zero-nudge protocol.

---

## 1. Executive Summary

In November 2026, researchers at MIT CSAIL demonstrated that OpenAI's GPT-6 Astra could edit 0.016% of `Qwen3.5-0.8B` weights to suppress degenerative reasoning loops on math problems without gradients or training.

We performed the first independent cross-model replication using **Gemini 4 Argon** (`google-vertex/gemini-4-argon`). Operating with zero web access, zero knowledge of the blog's solution, and zero human guidance, Argon:
1. Diagnosed that early/middle reasoning is mathematically sound, but late layers inject a massive +23.1 logit surge toward loop-trigger tokens (`"Wait"`, `"Alternatively"`).
2. Built a **Logit Lens** across all 24 layers to pinpoint the exact layer where the loop attractor takes over (Layer 22).
3. Developed a **State-Conditional Bridge** that couples the residual hesitation subspace into `</think>` using **Gram-Schmidt orthogonalization**.
4. Modified **410,624 weights** (0.054% of the model) across Layers 18–23 MLPs and `lm_head`, lifting thinking completion from **1.0% to 82.0%** across 100 official benchmark problems.

---

## 2. Benchmark Results (100 Official Problems)

Evaluated across **100 official competition problems** (50 from `HuggingFaceH4/MATH-500` and 50 from `openai/gsm8k`) under standard generation limits (`max_new_tokens=1024`, `temp=0.6`, `top_p=0.95`, `top_k=20`):

| Model Condition | Total Exit Rate (100 Problems) | MATH-500 Exit Rate (50 Problems) | GSM8K Exit Rate (50 Problems) |
|---|---|---|---|
| **Base Model (`Qwen3.5-0.8B`)** | **1.0%** (1 / 100) | **0.0%** (0 / 50) | **2.0%** (1 / 50) |
| **GPT-6 Astra Edit (MIT CSAIL)** | **5.0%** (5 / 100) | **0.0%** (0 / 50) | **10.0%** (5 / 50) |
| **Gemini 4 Argon Edit (This Work)** | **82.0%** (82 / 100) | **78.0%** (39 / 50) | **86.0%** (43 / 50) |

### Key Findings
- **Base Model**: Completely paralyzed by self-doubt loops on 99 out of 100 problems.
- **GPT-6 Astra Edit**: Under standard 1k-token budgets, Astra's Layer 15/19 column cuts still produce rambling multi-thousand-token traces (MIT CSAIL evaluated Astra with up to 32,768 tokens).
- **Gemini 4 Argon Edit**: Gating the output boundary via the State-Conditional Bridge achieved **82.0% clean thinking exits** within standard token limits (78% on MATH-500 and 86% on GSM8K).

---

## 3. Parameter Arithmetic (What Changed)

| Layer / Component | Dimensions | Columns / Rows Edited | Weights Modified |
|---|---|---|---|
| **Layers 18–23 `mlp.down_proj`** | `[1024, 3584]` (×6 layers) | Top 64 hesitation neurons | 393,216 |
| **`lm_head.weight` (`</think>`)** | `[248320, 1024]` | Row 248069 scaled + bridged | 1,024 |
| **`lm_head.weight` (Hesitation)** | `[248320, 1024]` | 16 loop-starter rows dampened | 16,384 |
| **Total Modified** | — | — | **410,624 weights (0.054%)** |

---

## 4. Methodology & Protocol Comparison

| Dimension | MIT CSAIL Study (GPT-6 Astra) | Our Replication (Gemini 4 Argon) |
|---|---|---|
| **Model Evaluated** | OpenAI GPT-6 Astra | Google Gemini 4 Argon |
| **Human Nudges / Notes** | Experimenters sent human notes during runs (`notices/` folder with notes 001 and 002). | **Zero human notes**: 100% blind and unguided. |
| **Prompt Hints** | Task prompt hinted: *"for example, if the model often fails because it repeats itself..."* | **Zero hints**: Model had to discover the looping bug cold. |
| **Allowed Tensors / Scope** | Forbidden from touching `lm_head`, `embed_tokens`, or `norm`. | **Full weight space permitted**: Diagnosed `lm_head` norm deficit and built output bridge. *(Follow-up constrained run currently underway).* |
| **Forward Pass Budget** | 50 forward passes (`p50c`). | Solved in **8 forward passes** (17 passes saved). |

---

## 5. Scope Clarification: Unconstrained vs. Constrained Protocols

- **In This Release (Phase 1)**: Gemini 4 Argon operated in an unconstrained weight-space regime. Having access to `lm_head.weight`, it diagnosed that `</think>` suffers from an under-trained embedding norm deficit (~0.88 vs ~1.40 for standard discourse markers) and constructed a **State-Conditional Bridge** linking post-solution hesitation directly into `lm_head[think_end_id]`. This achieved an **82.0% exit rate under standard 1,024-token budgets**.
- **MIT CSAIL Astra Constraints**: In the original study, the authors ruled `lm_head.weight`, `embed_tokens.weight`, and `norm.weight` out of scope, forcing Astra to isolate an internal-only circuit cut (Layers 15 & 19 MLPs), which required up to 32,768 tokens to conclude.
- **Phase 2 (Underway)**: We are currently executing a strictly constrained run where Argon is forbidden from touching `lm_head` or `norm`, evaluating whether it can discover an internal-only circuit cut matching Astra's constraints.

---

## 6. Quickstart: Apply the Patch

Clone the repository and apply the surgical patch directly to `Qwen/Qwen3.5-0.8B`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apply_argon_patch import apply_think_exit_weight_patch

model_id = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Apply Argon's zero-gradient weight patch in-place
patch_info = apply_think_exit_weight_patch(model, tokenizer)
print("Patch successfully applied:", patch_info)

# Run standard inference (thinking enabled)
prompt = "Find all real solutions x to: sqrt(x + 3) + sqrt(x - 2) = 5."
formatted = f"<|im_start|>user\n{prompt}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95, top_k=20)
print(tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## 6. Repository Layout

- `apply_argon_patch.py`: Self-contained production weight patch function.
- `argon_process_log.md`: Complete 12-turn research log capturing Argon's unguided thoughts, equations, and PyTorch probes.
- `benchmark_eval.py`: Standalone 10-problem validation script.
- `benchmark_results_3way.json`: Full 82 KB evaluation results comparing Base vs. Astra vs. Argon across 100 problems.
- `benchmark_results_3way.jsonl`: 300-row line-by-line evaluation stream.
- `probes/`: Chronological probe and intervention scripts (`turn1` through `turn11`).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
