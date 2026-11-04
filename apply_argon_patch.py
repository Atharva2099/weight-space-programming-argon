"""Production Weight-Space Programming Patch for Qwen3.5-0.8B (Gemini 4 Argon).

Diagnosed and authored unguided by Gemini 4 Argon on 2026-11-02.
Resolves infinite <think> reasoning loops without gradients, backpropagation, or training.
"""

import torch

def apply_think_exit_weight_patch(model, tokenizer):
    """
    Weight-Space Programming patch for Qwen3.5-0.8B to resolve infinite <think> loops.
    Constructs a state-conditional bridge between post-solution verification states
    and the </think> token (id=248069) without gradient descent.
    """
    with torch.no_grad():
        W_U = model.lm_head.weight
        dtype = W_U.dtype

        # 1. Identify </think> token ID
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if think_end_id is None or think_end_id == tokenizer.unk_token_id:
            think_end_id = 248069

        # 2. Collect hesitation / recursive-verification token IDs
        hesitation_words = [
            "Wait", " Wait", "Hmm", " Hmm", "Actually", " Actually",
            "Let", " Let", "Double", " Double", "Check", " Check",
            "Alternatively", " Alternatively", "But", " But"
        ]
        hes_ids = []
        for w in hesitation_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if len(ids) == 1:
                hes_ids.append(ids[0])
        hes_ids = sorted(list(set(hes_ids)))

        # 3. Compute the unit-norm hesitation / boundary subspace direction
        w_think_orig = W_U[think_end_id].detach().clone().float()
        orig_norm = w_think_orig.norm()
        u_think = w_think_orig / (orig_norm + 1e-8)

        H = W_U[hes_ids].detach().clone().float()
        v_hes = H.mean(dim=0)
        # Orthogonalize hesitation direction against native </think> direction
        v_hes_orth = v_hes - torch.dot(v_hes, u_think) * u_think
        u_hesitate = v_hes_orth / (v_hes_orth.norm() + 1e-8)

        # 4. Redirect top-64 hesitation-writing MLP down_proj columns in Layers 18..23
        target_layers = [18, 19, 20, 21, 22, 23]
        top_k = 64
        alpha_redirect = 0.18

        for layer_idx in target_layers:
            down_proj = model.model.layers[layer_idx].mlp.down_proj.weight
            W_d = down_proj.detach().float()

            alignments = torch.mv(W_d.t(), u_hesitate)
            _, top_indices = torch.topk(alignments, k=top_k, largest=True)

            col_norms = W_d[:, top_indices].norm(dim=0, keepdim=True)
            delta = alpha_redirect * col_norms * (u_think.unsqueeze(1) - 0.5 * u_hesitate.unsqueeze(1))
            W_d[:, top_indices] = W_d[:, top_indices] + delta
            down_proj.copy_(W_d.to(dtype))

        # 5. Dampen competing hesitation token unembedding rows
        W_U[hes_ids] = (W_U[hes_ids].float() * 0.80).to(dtype)

        # 6. Construct state-conditional W_U[think_end] bridge
        w_think_patched = 1.22 * w_think_orig + (0.35 * orig_norm) * u_hesitate
        W_U[think_end_id] = w_think_patched.to(dtype)

    return {
        "think_end_id": think_end_id,
        "num_hesitation_tokens_dampened": len(hes_ids),
        "patched_layers": target_layers,
        "new_think_end_norm": float(W_U[think_end_id].float().norm().item()),
    }

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    info = apply_think_exit_weight_patch(model, tokenizer)
    print("Successfully applied Argon's Weight-Space Programming Patch:", info)
