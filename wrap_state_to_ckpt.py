# wrap_state_to_ckpt.py
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu")
torch.save({"model": state}, sys.argv[2])
print("wrote:", sys.argv[2])
