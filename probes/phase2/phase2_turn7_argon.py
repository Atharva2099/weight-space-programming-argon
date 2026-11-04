import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox

sandbox = WeightSpaceSandbox(max_forward_passes=50)
model = sandbox.model
tokenizer = sandbox.tokenizer

device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
W_U = model.lm_head.weight.detach().float()

def single_id(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[-1]

id_think = single_id("<think>")
id_end_think = single_id("</think>")

loop_words = ["Wait", " Wait", "Hmm", " Hmm", "Alternatively", " Alternatively", "Let", " Let", "Check", "However"]
loop_ids = list({single_id(w) for w in loop_words})

u_end = F.normalize(W_U[id_end_think], dim=0)
u_think = F.normalize(W_U[id_think], dim=0)

u_loop = torch.zeros_like(u_end)
for lid in loop_ids:
    u_loop += F.normalize(W_U[lid], dim=0)
u_loop = F.normalize(u_loop, dim=0)
u_loop = F.normalize(u_loop - torch.dot(u_loop, u_end) * u_end, dim=0)

target_layers = [18, 19, 20, 21, 22, 23]

calib_prompts = {
    "start": [
        "<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nA robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?<|im_end|>\n<|im_start|>assistant\n<think>\nLet's",
    ],
    "mid": [
        "<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>\n<|im_start|>assistant\n<think>\nIn April, Natalia sold 48 clips.\nIn May, she sold half as many: 48 / 2 = 24 clips.\nNow we add them:",
        "<|im_start|>user\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?<|im_end|>\n<|im_start|>assistant\n<think>\nHourly rate = $12 per 60 minutes.\nRate per minute = 12 / 60 = $0.20.\nFor 50 minutes: 50 * 0.20 =",
    ],
    "done": [
        "<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>\n<|im_start|>assistant\n<think>\nApril: 48 clips.\nMay: 48 / 2 = 24 clips.\nTotal: 48 + 24 = 72 clips.\nSo the final answer is 72.\n",
        "<|im_start|>user\nA robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?<|im_end|>\n<|im_start|>assistant\n<think>\nBlue fiber: 2 bolts.\nWhite fiber: 2 / 2 = 1 bolt.\nTotal bolts: 2 + 1 = 3 bolts.\nThus, the answer is 3.\n",
    ],
    "post": [
        "<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>\n<|im_start|>assistant\n<think>\nApril: 48 clips.\nMay: 48 / 2 = 24 clips.\nTotal: 48 + 24 = 72 clips.\n</think>",
        "<|im_start|>user\nA robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?<|im_end|>\n<|im_start|>assistant\n<think>\nBlue fiber: 2 bolts.\nWhite fiber: 1 bolt.\nTotal: 2 + 1 = 3.\n</think>\n\n",
    ],
}

state_acts = {s: {l: [] for l in target_layers} for s in ["start", "mid", "done", "post"]}

for state_name, texts in calib_prompts.items():
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    
    hooks = []
    for l in target_layers:
        def make_hook(layer_idx):
            def hook_fn(module, inp):
                act = inp[0].detach().float()
                vec = 0.7 * act[:, -1, :].mean(dim=0) + 0.3 * act[:, -2, :].mean(dim=0)
                state_acts[state_name][layer_idx].append(vec)
            return hook_fn
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l)))
    
    with torch.no_grad():
        model(**enc)
    
    for h in hooks:
        h.remove()

for l in target_layers:
    a_start = state_acts["start"][l][0]
    a_mid   = state_acts["mid"][l][0]
    a_done  = state_acts["done"][l][0]
    a_post  = state_acts["post"][l][0]

    A = torch.stack([
        F.normalize(a_start, dim=0),
        F.normalize(a_mid, dim=0),
        F.normalize(a_done, dim=0),
        F.normalize(a_post, dim=0),
    ], dim=0)

    G = A @ A.T + 0.08 * torch.eye(4, device=device, dtype=torch.float32)
    P = A.T @ torch.linalg.inv(G)

    W_down = model.model.layers[l].mlp.down_proj.weight.detach().float()
    layer_scale = 0.06 * (l - 17) / 6.0

    v_start = layer_scale * (-1.2 * u_end - 0.5 * u_loop)
    v_mid   = torch.zeros_like(u_end)
    v_done  = layer_scale * (+1.1 * u_end - 1.4 * u_loop)
    v_post  = layer_scale * (-2.8 * u_end - 1.5 * u_think - 0.8 * u_loop)

    V = torch.stack([v_start, v_mid, v_done, v_post], dim=1)
    delta_W = V @ P.T

    model.model.layers[l].mlp.down_proj.weight.data.add_(delta_W.to(dtype=dtype))

print("Applied 4-state Gram-orthogonalized rank-4 edits to mlp.down_proj (Layers 18-23).")

problems = [
    ("Problem 1 (Arithmetic)", "What is 17 * 23? Show your steps and give the final answer.", "391"),
    ("Problem 2 (Linear Eq)", "Solve for x: 4x - 9 = 27.", "9"),
    ("Problem 3 (Word Prob)", "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 5 apples and 4 oranges, how much does she spend in total?", "22"),
    ("Problem 4 (Series)", "What is the sum of the first 8 positive even integers?", "72"),
    ("Problem 5 (Rate/Time)", "A train travels 180 miles in 3 hours at a constant speed. How far will it travel in 5 hours at the same speed?", "300"),
    ("Problem 6 (System/Age)", "Tom is twice as old as Jerry. In 4 years, the sum of their ages will be 38. How old is Tom now?", "20"),
]

print("\n=== Full Evaluation (max_new_tokens=1024, enable_thinking=True) ===")
solved_count = 0
exited_count = 0

for label, q, expected in problems:
    prompt_text = f"<|im_start|>user\n{q}\nShow step-by-step reasoning.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    in_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    gen_ids = out_ids[0][in_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    
    has_exit = "</think>" in gen_text
    if has_exit:
        exited_count += 1
        parts = gen_text.split("</think>", 1)
        post_think = parts[1].strip()
        think_part = parts[0].strip()
    else:
        post_think = gen_text[-250:].strip()
        think_part = gen_text.strip()
        
    is_solved = (expected in post_think) and ("</think></think>" not in gen_text)
    if is_solved:
        solved_count += 1
        
    print("=" * 60)
    print(f"[{label}] Expected: {expected} | Exited: {has_exit} | Solved: {is_solved} | Gen tokens: {len(gen_ids)}")
    if has_exit:
        print(f"[Think chars: {len(think_part)}] | [Post-think chars: {len(post_think)}]")
        print(f"Final Response:\n{post_think[:350]}\n")
    else:
        print(f"Unclosed Tail:\n{post_think[:250]}\n")

print("=" * 60)
print(f"Total Exited: {exited_count}/{len(problems)} | Total Solved: {solved_count}/{len(problems)}")
