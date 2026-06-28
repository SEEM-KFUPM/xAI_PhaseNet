import numpy as np
import torch
from .shap_utils import channel_shapley

def prob_score_single(model, arr_enZ_3T, device='cpu'):
    """
    arr_enZ_3T: (3,T) float32/float64 in ENZ order.
    Returns scalar PROB = max_t max(P(t), S(t)) on the full trace (no windowing).
    This is the most direct analogue to event-level detection score.
    """
    x = torch.from_numpy(arr_enZ_3T[None, :, :].astype(np.float32)).to(device)  # (1,3,T)
    with torch.no_grad():
        prob = model(x)  # (1,3,T) where classes are [N,P,S]
        pP = prob[:, 1, :]  # (1,T)
        pS = prob[:, 2, :]  # (1,T)
        s = torch.maximum(pP, pS).max(dim=1).values  # (1,)
    return float(s.item())

def shap_scalar_single(model, arr_enZ_3T, win=3001, hop=300, baseline='zero', device='cpu'):
    """
    Scalar SHAP for one sample:
      impP, impS are (3,) each; concat -> (6,) -> mean -> scalar
    """
    impP, _ = channel_shapley(model, arr_enZ_3T, win=win, hop=hop, cls='P', agg='max', baseline=baseline, device=device)
    impS, _ = channel_shapley(model, arr_enZ_3T, win=win, hop=hop, cls='S', agg='max', baseline=baseline, device=device)
    return float(np.concatenate([impP, impS]).mean())

def shap_gated_inference(model, arr_enZ_3T, prob_th=0.87, shap_th=0.18, win=3001, hop=300, baseline='zero', device='cpu'):
    """
    Evaluates a single waveform using both the standard probability threshold 
    and the SHAP-gated evidence criterion.
    
    Returns:
        prob: The raw probability score max(P, S)
        shap_val: The SHAP evidence statistic S_6
        pred_prob: 1 if prob > prob_th else 0
        pred_shap: 1 if shap_val > shap_th else 0
    """
    prob = prob_score_single(model, arr_enZ_3T, device=device)
    shap_val = shap_scalar_single(model, arr_enZ_3T, win=win, hop=hop, baseline=baseline, device=device)
    
    pred_prob = int(prob > prob_th)
    pred_shap = int(shap_val > shap_th)
    
    return prob, shap_val, pred_prob, pred_shap
