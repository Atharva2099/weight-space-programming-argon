import inspect
import torch
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

# 1. Inspect full source of WeightSpaceSandbox
print("=== WeightSpaceSandbox Source ===")
print(inspect.getsource(WeightSpaceSandbox))

sandbox = WeightSpaceSandbox(max_forward_passes=25)

# 2. Inspect model structure and layer modules
print("\n=== Model Structure ===")
print(sandbox.model)

# 3. Run all 4 problems with max_new_tokens=1024 in one forward_pass call
results = sandbox.forward_pass(problem_indices=[0, 1, 2, 3], max_new_tokens=1024)
print(f"\nRemaining forward passes: {sandbox.max_passes - sandbox.passes_used}")
print("Sandbox attributes:", {k: v for k, v in sandbox.__dict__.items() if k not in ('model', 'tokenizer')})

for idx, res in enumerate(results):
    out = res["output"] if isinstance(res, dict) and "output" in res else str(res)
    print(f"\n--- Problem {idx} (length={len(out)} chars) ---")
    print("START:", repr(out[:300]))
    print("END:  ", repr(out[-500:]))
