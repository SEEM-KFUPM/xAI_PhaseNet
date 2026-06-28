import torch
import numpy as np

def compute_gradcam(model, signal_batch, target_class_idx):
    """
    Computes Grad-CAM localization map for a given signal batch.
    
    Args:
        model: The PhaseNet model (should have get_activations, get_gradient, etc.)
        signal_batch: Input tensor of shape (1, 3, T)
        target_class_idx: Int (e.g., 0 for Noise, 1 for P, 2 for S) or None.
            If None, overall_score = sum over Noise (as in original notebook).
            Wait, original notebook used 0 for Noise. 
    
    Returns:
        localization_map: 1D numpy array of length T
    """
    model.eval()
    
    # Forward pass
    logits = model(signal_batch)
    feat_maps = model.get_activations(signal_batch).detach()
    
    # Target score
    # In original notebook: overall_score = logits[:, 0, :].sum() # sum over Noise channel
    # To be general, we use target_class_idx. If we want to focus on an event, we might want P or S.
    # Actually, the notebook did: `overall_score = logits[:, 0, :].sum()`
    # Let's make it flexible
    overall_score = logits[:, target_class_idx, :].sum()
    
    model.zero_grad()
    overall_score.backward(retain_graph=True)
    
    gradients = model.get_gradient()
    
    # Pool gradients over time
    dim2 = [0, 2] # pool over batch and time
    pooled_gradients = torch.mean(gradients, dim=dim2)
    
    feat_maps22 = feat_maps.clone()
    for i in range(feat_maps22.size(1)):
        feat_maps22[:, i, :] *= pooled_gradients[i]
        
    localization_map = torch.sum(feat_maps22, dim=1).squeeze()
    localization_map = localization_map.cpu().numpy()
    
    # ReLU
    localization_map = np.maximum(localization_map, 0)
    
    # Normalize
    if np.max(localization_map) > 0:
        localization_map = localization_map / np.max(localization_map)
        
    return localization_map, feat_maps, gradients
