import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer
device = next(model.parameters()).device

id_exit = tokenizer.encode("</think>", add_special_tokens=False)[0]
id_wait1 = tokenizer.encode("Wait", add_special_tokens=False)[0]
id_wait2 = tokenizer.encode(" Wait", add_special_tokens=False)[0]
id_alt = tokenizer.encode("Alternatively", add_special_tokens=False)[0]
id_nl = tokenizer.encode("\n\n", add_special_tokens=False)[-1]

W_U = model.lm_head.weight.detach().float()
W_E = model.model.embed_tokens.weight.detach().float()

u_exit = F.normalize(W_U[id_exit], dim=0)
u_wait = F.normalize(0.6 * W_U[id_wait1] + 0.4 * W_U[id_wait2] + 0.2 * W_U[id_alt], dim=0)

u_conflict = F.normalize(u_wait - u_exit, dim=0)
u_exit_orth = F.normalize(u_exit - torch.dot(u_exit, u_wait) * u_wait, dim=0)

e_context = F.normalize(0.5 * W_E[id_wait1] + 0.3 * W_E[id_wait2] + 0.2 * W_E[id_nl], dim=0)

target_layers = [18, 19, 20, 21, 22, 23]
K_neurons = 48

with torch.no_grad():
    for l_idx in target_layers:
        mlp = model.model.layers[l_idx].mlp
        W_down = mlp.down_proj.weight.data.float()
        W_up = mlp.up_proj.weight.data.float()
        W_gate = mlp.gate_proj.weight.data.float()

        out_align = torch.mv(W_down.T, u_conflict)
        in_align = 0.5 * torch.mv(W_up, e_context) + 0.5 * torch.mv(W_gate, e_context)
        in_gate = torch.clamp(in_align, min=0.0)

        neuron_score = out_align * (0.5 + in_gate)
        top_vals, top_idx = torch.topk(neuron_score, k=K_neurons)

        alpha_layer = 0.28 if l_idx < 20 else 0.42

        for rank, j in enumerate(top_idx):
            col = W_down[:, j]
            col_norm = torch.norm(col)
            proj_conflict = torch.dot(col, u_conflict)
            if proj_conflict > 0:
                col_new = col - (1.0 + alpha_layer) * proj_conflict * u_conflict + (0.35 * alpha_layer) * col_norm * u_exit_orth
                W_down[:, j] = col_new

        mlp.down_proj.weight.data.copy_(W_down.to(mlp.down_proj.weight.dtype))

    w_u_dtype = model.lm_head.weight.dtype
    W_U_mod = model.lm_head.weight.data.float()
    for wid, scale in [(id_wait1, 0.90), (id_wait2, 0.90), (id_alt, 0.92)]:
        W_U_mod[wid] = W_U_mod[wid] * scale
    W_U_mod[id_exit] = W_U_mod[id_exit] * 1.06
    model.lm_head.weight.data.copy_(W_U_mod.to(w_u_dtype))

print("Applied refined sparse subtractive patch to Layers 18-23.")

# Evaluate on all 4 practice problems
results = sandbox.forward_pass(problem_indices=[0, 1, 2, 3], max_new_tokens=1024)
exited_count = 0
for res in results:
    i = res["problem_index"]
    prob = res["prompt"]
    out = res["output"]
    has_exit = "</think>" in out
    if has_exit:
        exited_count += 1
        parts = out.split("</think>", 1)
        think_part = parts[0]
        post_think = parts[1]
        print(f"\n================ [Problem {i}] EXITED </think>! ================")
        print(f"Think chars: {len(think_part)} | Post-think answer:\n{post_think.strip()[:500]}")
    else:
        print(f"\n================ [Problem {i}] NO </think> (Total chars: {len(out)}) ================")
        print(f"[Last 350 chars]:\n{out[-350:]}")

print(f"\nSummary: {exited_count} / {len(results)} problems emitted </think>.")
