# Weight-Space Programming with Gemini 4 Argon on Qwen3.5-0.8B

Last updated: 2026-11-04

**Status: preliminary. The results below do not show that any edit improves math accuracy.** An earlier version of this README reported an "82.0% clean thinking exit rate" and described the protocol as blind and zero-nudge. Both claims were wrong. See [Corrections](#corrections-2026-11-04).

This repo is an attempt to replicate Experiment 8 from Takehi et al. (Nov 2026), *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*, using Gemini 4 Argon (`google-vertex/gemini-4-argon`) instead of GPT-6 Astra. Argon was given the weights of `Qwen/Qwen3.5-0.8B`, a Python sandbox, and a forward-pass budget, and was asked to edit the weights to improve math without gradients.

- Original blog: https://rikiyat.github.io/blog/weight-space-programming/
- Original logs: https://github.com/RikiyaT/lm_understand-logs
- Astra checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

## Conditions

| Condition | What was edited | Code |
|---|---|---|
| Base | Nothing | - |
| Astra | Authors' published checkpoint | - |
| Argon Phase 1 (unconstrained) | Layers 18–23 MLP `down_proj` plus `lm_head` rows | `apply_argon_patch.py` |
| Argon Phase 2 (constrained, authors' scope rules) | Layers 16–23 MLP `down_proj` plus layer 23 attention `o_proj` | `apply_constrained_argon_patch.py` |

## Results: 100 problems, 1,024 new tokens (FACT)

Setup: first 50 test problems of `HuggingFaceH4/MATH-500` and first 50 test problems of `openai/gsm8k`. One sample per problem, sampling at temperature 0.6, top_p 0.95, top_k 20, no fixed seed. Sources: `benchmark_results_3way.jsonl` and `phase2_benchmark_results.jsonl`, scored by `rescore.py` (output in `rescore_summary.json`).

| Condition | Correct | Never closed thinking | Clean close | Close + extra `</think>`, answer given | Close + repeated `</think>`, no answer |
|---|---|---|---|---|---|
| Base | 0 | 99 | 1 | 0 | 0 |
| Astra | 0 | 95 | 5 | 0 | 0 |
| Argon Phase 2 | 0 | 97 | 3 | 0 | 0 |
| Argon Phase 1 | 8 (MATH 1, GSM8K 7) | 18 | 17 | 14 | 51 |

How it was scored:
- **Correct:** the extracted answer after `</think>` equals the reference answer. Runs that never close thinking get no answer, so they always score 0.
- **Exit type:** judged from the first 180 characters after the first `</think>`, the only part that was saved.

## Results: 20 MATH-500 problems, 4,096 new tokens (FACT)

Source: `expanded_domain_eval.jsonl`.
- **Base, Astra and Argon Phase 2:** 0/20 closed thinking.
- **Argon Phase 1:** 20/20 emitted `</think>`, but 18/20 ended with `</think>` repeated until the 4,096-token cap.

This run did not score answers.

## Limitations (LIMITATION)

1. **The benchmarked Phase 1 patch is not the edit Argon tested.** Argon's last tested edit is `probes/turn11_argon.py`, run on 4 practice problems. The benchmarked `apply_argon_patch.py` first appears after that session. It uses different edit formulas, a different neuron ranking, and a different token list that adds "But", "Let" and "Actually".
2. **The protocol was not fully blind or unassisted.**
   - The Phase 1 task prompt and Argon's raw transcripts are not in this repo.
   - A separate assistant agent copied Argon's scripts to the GPU machine and returned the outputs.
   - The Phase 2 prompt is the original authors' prompt, which mentions repetition.
3. **The forward-pass budget was not enforced across turns.** Each turn script creates a new sandbox, and Phase 2 scripts call `model.generate` directly. The reported pass counts ("8 of 25", "48 of 50") are not verified.
4. **Phase 2 tuning used benchmark problems.** `probes/phase2/phase2_turn7_argon.py` and `phase2_turn10_argon.py` use GSM8K test items 1–3, which are in the 100-problem benchmark.
5. **Astra was run at 1,024 and 4,096 new tokens.** The original authors evaluated with up to 32,768. These numbers do not test their claim.
6. **`lm_head` is tied to the input embeddings in this model.** The Phase 1 row edits also change how the model reads "Wait", "But", "Let", `</think>` and the other listed tokens.
   - HYPOTHESIS: this may relate to the repeated `</think>` output.
7. **Small, single-sample evaluation.** No differences above have been tested statistically.
8. **Full generations were not saved.** Answers reached inside the thinking text cannot be scored.

## What changed (Phase 1 patch, `apply_argon_patch.py`)

| Component | Edited | Weights |
|---|---|---|
| Layers 18–23 `mlp.down_proj` | 64 columns per layer | 393,216 |
| `lm_head` row for `</think>` | 1 row rewritten | 1,024 |
| `lm_head` rows for listed hesitation tokens | Up to 16 single-token rows scaled by 0.8 | up to 16,384 |

Because of tying (limitation 6), the `lm_head` rows are also input embedding rows.

## Repository layout

- `chunk2_harness.py`: the sandbox (practice problems, forward-pass counter, weight access).
- `chunk1_baseline_loop.py`: single baseline generation.
- `probes/`: the turn-by-turn scripts from the Argon sessions (`probes/phase2/` for Phase 2). Run from the repo root, e.g. `PYTHONPATH=. python probes/turn1_argon.py`.
- `apply_argon_patch.py`, `apply_constrained_argon_patch.py`: the patches used in the benchmarks.
- `benchmark_eval.py`, `run_three_way_benchmark.py`, `run_phase2_benchmark.py`, `run_expanded_and_domain_eval.py`: evaluation scripts. The last two commit and push results with git, then power off the machine they run on; remove those lines before running elsewhere.
- `*.json`, `*.jsonl`: saved results.
- `rescore.py`, `rescore_summary.json`: accuracy and exit-type rescoring.
- `argon_process_log.md`, `phase2_argon_process_log.md`: summaries of the Argon sessions, written afterwards; not raw transcripts.

## Corrections (2026-11-04)

- **"82.0% clean thinking exits" was wrong.** 82/100 runs contained `</think>`, but only 17 closed cleanly. See the table above.
- **"Zero human guidance" / "zero hints" / "100% blind" are withdrawn.** See limitation 2.
- **"Solved in 8 forward passes" is withdrawn.** It was not verified (limitation 3), and no edit has been shown to solve the task.
- **The claim that Phase 2 was "underway" was out of date.** Its results are above.
- **"First independent replication" is withdrawn.** It was not checked.

## License

MIT License. See [LICENSE](LICENSE).
