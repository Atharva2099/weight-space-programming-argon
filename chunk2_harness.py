"""Chunk 2: Sandboxed environment for weight-space programming."""

from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PRACTICE_PROBLEMS = [
    "Find all real solutions x to the equation: sqrt(x + 3) + sqrt(x - 2) = 5.",
    "If f(x) = 2x + 3 and g(x) = x^2 - 1, find the value of f(g(3)) - g(f(2)).",
    "A train travels 120 miles at 40 mph, then returns the same distance at 60 mph. What is its average speed for the round trip?",
    "Find the sum of all integers n such that (n - 2) divides (n^2 + 10).",
]

class WeightSpaceSandbox:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-0.8B", max_forward_passes: int = 25):
        print(f"Initializing sandbox for {model_id} on CUDA...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.max_passes = max_forward_passes
        self.passes_used = 0

    def forward_pass(self, problem_indices: List[int], max_new_tokens: int = 512) -> List[Dict[str, Any]]:
        """Run a forward pass on selected practice problems. Counts toward budget."""
        if self.passes_used >= self.max_passes:
            raise RuntimeError(f"Budget exhausted: {self.passes_used}/{self.max_passes} forward passes used.")
        
        self.passes_used += 1
        results = []

        for idx in problem_indices:
            prompt = PRACTICE_PROBLEMS[idx % len(PRACTICE_PROBLEMS)]
            # Format with thinking tokens enabled
            formatted = f"<|im_start|>user\n{prompt}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
            inputs = self.tokenizer([formatted], return_tensors="pt").to("cuda")

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                )

            gen_text = self.tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append({"problem_index": idx, "prompt": prompt, "output": gen_text})

        return results

    def inspect_weights(self, layer_idx: int, component: str) -> torch.Tensor:
        """Inspect a specific weight tensor without gradients."""
        layer = self.model.model.layers[layer_idx]
        if component == "mlp_down":
            return layer.mlp.down_proj.weight.detach().clone()
        elif component == "mlp_gate":
            return layer.mlp.gate_proj.weight.detach().clone()
        elif component == "mlp_up":
            return layer.mlp.up_proj.weight.detach().clone()
        else:
            raise ValueError(f"Unknown component: {component}")

if __name__ == "__main__":
    sandbox = WeightSpaceSandbox(max_forward_passes=5)
    print(f"Testing sandbox forward pass (1/{sandbox.max_passes})...")
    res = sandbox.forward_pass([0], max_new_tokens=256)
    print(f"Pass used: {sandbox.passes_used}/{sandbox.max_passes}")
    print("Thinking snippet:\n", res[0]["output"][:300])
