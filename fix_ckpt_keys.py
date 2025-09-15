# fix_ckpt_keys.py
import sys, torch, os

inp = sys.argv[1]           # e.g., .\output\run_1757458008\best_model_state.pt  OR best_model_orig.pth
out = sys.argv[2]           # e.g., .\output\run_1757458008\best_model.pth

obj = torch.load(inp, map_location="cpu")

# Accept either a raw state_dict or {"model": state_dict}
state = obj
if isinstance(obj, dict) and "model" in obj and all(isinstance(k, str) for k in obj["model"]):
    state = obj["model"]

# Strip any 'module.' prefix and rename old keys -> new keys expected by runtime
new_state = {}
for k, v in state.items():
    k2 = k
    if k2.startswith("module."):        # if saved with DDP/DataParallel
        k2 = k2[len("module."):]
    k2 = k2.replace("bert_head", "headB")
    k2 = k2.replace("roberta_head", "headR")
    k2 = k2.replace("weight_bert", "wB")
    k2 = k2.replace("weight_roberta", "wR")
    new_state[k2] = v

# Wrap in the format runtime expects: {"model": state_dict}
torch.save({"model": new_state}, out)
print(f"Wrote {out} with {len(new_state)} params.")
