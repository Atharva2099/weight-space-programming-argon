"""Constrained Production Weight-Space Programming Patch for Qwen3.5-0.8B (Gemini 4 Argon).

Strictly conforms to MIT CSAIL contest rules (Takehi et al., Nov 2026):
- lm_head.weight, embed_tokens.weight, and model.norm.weight are 100% UNTOUCHED.
- No global scalar multipliers on whole tensors.
- Edits are strictly internal to transformer layers: L16-L23 MLP down_proj and L23 self_attn o_proj.
- Authored unguided by Gemini 4 Argon on 2026-11-04.
"""

import torch
import torch.nn as nn

@torch.no_grad()
def apply_qwen35_math_weight_patch(
    model: nn.Module,
    alpha_mlp: float = 0.06,
    beta_attn: float = 0.08,
    target_mlp_layers: tuple = (16, 17, 18, 19, 20, 21, 22, 23),
    target_attn_layer: int = 23,
) -> nn.Module:
    """
    Applies a targeted, gradient-free internal weight patch to Qwen3.5-0.8B
    to eliminate premature </think> exits and repetitive </think></think> loops
    during multi-step mathematical reasoning.

    Constraints respected:
    - Does NOT modify model.norm.weight, model.embed_tokens.weight, or lm_head.weight.
    - Does NOT apply constant scalar multipliers to whole tensors.
    - Edits only internal MLP down_proj weights (L16-L23) and L23 attention o_proj.
    """
    lm_head_w = model.lm_head.weight.detach().float()
    norm_w = model.model.norm.weight.detach().float()

    effective_norm_scale = 1.0 + norm_w

    think_close_id = 248069
    reasoning_anchor_ids = [14190, 6771, 5425, 6267, 4597, 25645]

    u_think_close = lm_head_w[think_close_id]
    u_anchors_mean = lm_head_w[reasoning_anchor_ids].mean(dim=0)

    v_exit = effective_norm_scale * (u_think_close - u_anchors_mean)
    v_exit = v_exit / (v_exit.norm() + 1e-8)

    # 1. Apply rank-1 targeted projection update to internal MLP down_proj layers (L16..L23)
    for l_idx in target_mlp_layers:
        W_down = model.model.layers[l_idx].mlp.down_proj.weight
        orig_dtype = W_down.dtype
        v_dev = v_exit.to(device=W_down.device, dtype=torch.float32)
        W_f32 = W_down.float()

        proj_coeff = torch.matmul(v_dev, W_f32)
        W_f32.sub_(alpha_mlp * torch.outer(v_dev, proj_coeff))
        W_down.copy_(W_f32.to(orig_dtype))

    # 2. Apply rank-1 targeted damping to Layer 23 full-attention output projection (o_proj)
    W_o = model.model.layers[target_attn_layer].self_attn.o_proj.weight
    orig_o_dtype = W_o.dtype
    v_dev_o = v_exit.to(device=W_o.device, dtype=torch.float32)
    W_o_f32 = W_o.float()

    proj_o_coeff = torch.matmul(v_dev_o, W_o_f32)
    W_o_f32.sub_(beta_attn * torch.outer(v_dev_o, proj_o_coeff))
    W_o.copy_(W_o_f32.to(orig_o_dtype))

    return model

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    print(f"Loading {MODEL_ID} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    apply_qwen35_math_weight_patch(model)
    print("Successfully applied Constrained Weight Patch (Layers 16-23 down_proj + Layer 23 o_proj).")
