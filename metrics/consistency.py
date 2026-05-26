import torch

def overlap_consistency(values_a, idx_a, values_b, idx_b):
    m = {int(i):k for k,i in enumerate(idx_a.tolist())}
    errs=[]
    for j,i in enumerate(idx_b.tolist()):
        i=int(i)
        if i in m:
            errs.append((values_a[m[i]]-values_b[j]).abs().mean())
    if not errs:
        return torch.tensor(0.0)
    return torch.stack(errs).mean()
