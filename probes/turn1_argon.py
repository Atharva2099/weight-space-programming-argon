import inspect
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

print("Number of practice problems:", len(PRACTICE_PROBLEMS))
for i, p in enumerate(PRACTICE_PROBLEMS[:5]):
    print(f"Problem {i}: {p}")

print("\nforward_pass source:")
print(inspect.getsource(WeightSpaceSandbox.forward_pass))

sandbox = WeightSpaceSandbox(max_forward_passes=25)
print("\nModel type:", type(sandbox.model))
print("Num layers:", len(sandbox.model.model.layers) if hasattr(sandbox.model, "model") else "unknown")

# Run baseline on first 3 problems to observe the failure mode
results = sandbox.forward_pass(problem_indices=[0, 1, 2], max_new_tokens=512)
print(f"\nRemaining forward passes: {25 - getattr(sandbox, 'forward_passes_used', getattr(sandbox, 'passes_used', -1))}")
for r in results:
    print("=" * 60)
    print(f"Problem {r['problem_index']} Prompt:\n{r['prompt']}")
    print(f"Output:\n{r['output'][:1000]}")
