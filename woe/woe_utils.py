try:
    import torch
except:
    print("no pytorch")


def int2hot(idx, n=None):
    if type(idx) is int:
        idx = torch.tensor([idx])
    if idx.dim == 0:
        idx = idx.view(-1, 1)
    if not n:
        n = idx.max() + 1
    return torch.zeros(len(idx), n).scatter_(1, idx.unsqueeze(1), 1.0)
