import torch
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=50)

print("Number of practice problems:", len(PRACTICE_PROBLEMS))
for i in range(min(5, len(PRACTICE_PROBLEMS))):
    print(f"--- Problem {i} ---")
    print(PRACTICE_PROBLEMS[i])

# Inspect model structure and parameter names for one linear-attention layer (layer 0) and one full-attention layer (layer 3)
model = sandbox.model
print("\nModel type:", type(model))
for name, param in model.named_parameters():
    if "layers.0." in name or "layers.3." in name or "layers.23." in name or "layers" not in name:
        print(f"{name}: {tuple(param.shape)} {param.dtype}")

# Run a small initial check on 4 problems to see how `forward_pass` works and what failures occur
res = sandbox.forward_pass([0, 1, 2, 3], max_new_tokens=1024)
print("\nForward pass return type and keys:", type(res))
if isinstance(res, dict):
    print("Keys:", res.keys())
elif isinstance(res, list):
    print("List len:", len(res), "Elem 0 type:", type(res[0]))
    if isinstance(res[0], dict):
        print("Elem 0 keys:", res[0].keys())
    for idx, item in enumerate(res):
        print(f"\n=== Output {idx} ===")
        print(str(item)[:1000])
