import inspect
import torch
import torch.nn.functional as F
from chunk2_harness import WeightSpaceSandbox, PRACTICE_PROBLEMS

sandbox = WeightSpaceSandbox(max_forward_passes=25)
model = sandbox.model
tokenizer = sandbox.tokenizer

prob0 = PRACTICE_PROBLEMS[0]
q_text = prob0["question"] if isinstance(prob0, dict) else str(prob0)

prompt_str = f"<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n<think>\n"

sample_trace = (
    prompt_str
    + "Let's solve this step by step.\n"
    + "First, let's read the problem carefully and compute the value.\n"
    + "So the answer is 42.\n\n"
    + "Wait, let me double check that.\n"
    + "Yes, the calculation is definitely correct.\n\n"
    + "Wait, let's check again."
)

input_ids = tokenizer.encode(sample_trace, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

logits = out.logits[0]
hidden_states = out.hidden_states

think_end_id = 248069
wait_id = tokenizer.encode("Wait", add_special_tokens=False)[0]

tokens_str = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
prompt_len = len(tokenizer.encode(prompt_str, add_special_tokens=False))

print(f"\nToken-by-token logits after prompt (think_end_id={think_end_id}, wait_id={wait_id}):")
for pos in range(max(0, prompt_len - 3), len(tokens_str)):
    top_val, top_idx = torch.max(logits[pos], dim=-1)
    top_tok = tokenizer.decode([top_idx.item()])
    l_end = logits[pos, think_end_id].item()
    l_wait = logits[pos, wait_id].item()
    print(f"Pos {pos:3d} | cur={repr(tokens_str[pos]):15s} -> top1={repr(top_tok):12s} ({top_val.item():6.2f}) | </think>={l_end:6.2f} | Wait={l_wait:6.2f}")

wait_positions = [pos for pos in range(prompt_len, len(tokens_str) - 1) if input_ids[0, pos + 1].item() == wait_id]
print("\nPositions right before 'Wait':", wait_positions)

if wait_positions:
    pos_c = wait_positions[0]
    pos_early = prompt_len
    print(f"\nLogit lens at Pos {pos_c} (right before 'Wait') vs Pos {pos_early} (start of <think>):")
    w_end = model.lm_head.weight[think_end_id].float()
    w_wait = model.lm_head.weight[wait_id].float()
    for l_idx, h_state in enumerate(hidden_states):
        h_c_norm = model.model.norm(h_state[0, pos_c : pos_c + 1]).squeeze(0).float()
        h_e_norm = model.model.norm(h_state[0, pos_early : pos_early + 1]).squeeze(0).float()
        logit_end_c = torch.dot(h_c_norm, w_end).item()
        logit_wait_c = torch.dot(h_c_norm, w_wait).item()
        logit_end_e = torch.dot(h_e_norm, w_end).item()
        print(f"  Layer {l_idx:2d}: PosC(</think>={logit_end_c:+6.2f}, Wait={logit_wait_c:+6.2f}, diff={logit_end_c-logit_wait_c:+6.2f}) | PosEarly(</think>={logit_end_e:+6.2f})")
