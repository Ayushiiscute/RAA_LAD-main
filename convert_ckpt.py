# convert_ckpt.py
import argparse, torch
from raa_lad import TrainingConfig  # just to register the class name

try:
    # PyTorch 2.6+ (no-op on older versions)
    from torch.serialization import add_safe_globals
    add_safe_globals([TrainingConfig])  # register the class so torch.load can unpickle
except Exception:
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to full checkpoint (.pth)")
    ap.add_argument("--out",  required=True, help="Path to write plain state_dict (.pt)")
    args = ap.parse_args()

    # IMPORTANT: weights_only=False so we actually unpickle the checkpoint once
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Try common keys; if none, assume it's already a state_dict
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    torch.save(state, args.out)
    print(f"✅ Saved state_dict to {args.out} ({len(state)} tensors)")

if __name__ == "__main__":
    main()
