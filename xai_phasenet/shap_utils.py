import numpy as np
import torch

def make_windows(arr, win=3001, hop=300):
    # arr: (3, T) in ENZ order
    T = arr.shape[1]
    starts = np.arange(0, T - win + 1, hop, dtype=int)
    return np.stack([arr[:, s:s+win] for s in starts], axis=0).astype(np.float32)  # (N,3,win)

def score(model, batch_np, cls='P', agg='max', device='cpu'):
    # batch_np: (N,3,L) -> scalar per window via P-prob aggregated over time
    x = torch.from_numpy(batch_np).to(device)
    with torch.no_grad():
        prob = model(x)                  # (N, 3, L) softmax over classes
        c = {'N':0,'P':1,'S':2}[cls]        
        pc = prob[:, c, :]                # (N, L)
        if   agg == 'max': s = pc.max(dim=1).values
        elif agg == 'mean': s = pc.mean(dim=1)
        elif agg == 'sum': s = pc.sum(dim=1)
        else: raise ValueError("agg ∈ {max, mean, sum}")
    return s.detach().cpu().numpy()       # (N,)

def mask_channels(x, keep, baseline='zero'):
    # x: (N,3,L), keep: e.g. [1,0,1] -> keep E,Z; mask N
    if baseline == 'zero':
        base = np.zeros_like(x)
    elif baseline == 'mean':
        base = x.mean(axis=2, keepdims=True).repeat(x.shape[2], axis=2)
    m = np.array(keep, dtype=x.dtype).reshape(1,3,1)
    return m * x + (1-m) * base

def channel_shapley(model, arr_enZ, win=3001, hop=300, cls='P', agg='max', baseline='zero', device='cpu'):
    """
    Computes component-wise Shapley values for the given 3C array.
    Returns:
        imp: Mean absolute Shapley value across windows for each component (shape: (3,))
        phi: Shapley values for each window and each component (shape: (N, 3))
    """
    X = make_windows(arr_enZ, win, hop)                # (N,3,win)
    masks = np.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1]
    ])
    vals = [score(model, mask_channels(X, m, baseline), cls, agg, device) for m in masks]  # each (N,)
    V = np.stack(vals, axis=1)  # (N,8)
    
    V000,V001,V010,V011,V100,V101,V110,V111 = [V[:,i] for i in range(8)]
    
    phi_E = (1/3)*(V100-V000) + (1/6)*(V110-V010) + (1/6)*(V101-V001) + (1/3)*(V111-V011)
    phi_N = (1/3)*(V010-V000) + (1/6)*(V110-V100) + (1/6)*(V011-V001) + (1/3)*(V111-V101)
    phi_Z = (1/3)*(V001-V000) + (1/6)*(V101-V100) + (1/6)*(V011-V010) + (1/3)*(V111-V110)
    
    phi = np.stack([phi_E, phi_N, phi_Z], axis=1)      # (N,3)
    imp = np.abs(phi).mean(axis=0)                     # (3,)
    
    return imp, phi
