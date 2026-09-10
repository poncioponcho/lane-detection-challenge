"""Uniform weight soup over N same-architecture checkpoints (UnLanedet format).

Usage: make_soup.py <out.pth> <ckpt1> <ckpt2> [...]
Averages the 'model' tensor dict entry-wise (floating tensors in fp32, then
saved fp32; FP16 conversion happens only at freeze time).  All checkpoints
must share identical keys — a mismatch aborts before writing.
"""
import sys
import torch

def main() -> None:
    out, paths = sys.argv[1], sys.argv[2:]
    assert len(paths) >= 2, "need >=2 checkpoints"
    acc = None
    keys_ref = None
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        sd = ck.get("model") or ck.get("state_dict")
        if sd is None:
            raise SystemExit(f"{p}: no 'model' key")
        if keys_ref is None:
            keys_ref = list(sd.keys())
            acc = {k: (v.clone().float() if torch.is_tensor(v) and v.is_floating_point() else v)
                   for k, v in sd.items()}
        else:
            if list(sd.keys()) != keys_ref:
                raise SystemExit(f"{p}: key mismatch vs first checkpoint")
            for k, v in sd.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    acc[k] += v.float()
    n = float(len(paths))
    for k in acc:
        if torch.is_tensor(acc[k]) and acc[k].is_floating_point():
            acc[k] /= n
    torch.save({"model": acc, "iteration": -1, "__soup__": {"members": paths, "n": len(paths)}}, out)
    import os
    print(f"soup of {len(paths)} -> {out} ({os.path.getsize(out)/1e6:.1f}MB)")

if __name__ == "__main__":
    main()
