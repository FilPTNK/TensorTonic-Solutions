import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x)
    ndim = x.ndim

    if ndim == 2:
        axis = 0
    elif ndim == 4:
        axis = (0,2,3)
        gamma = np.reshape(gamma, (1,-1,1,1))
        beta = np.reshape(beta, (1,-1,1,1))
    else:
        raise ValueError(f"Unsupported input dimension: {ndim}")
    
    mean = np.mean(x, axis = axis, keepdims = True)
    var = np.var(x, axis = axis, keepdims = True)
    normalized_x = (x-mean)/np.sqrt(var + eps)
    
    normalized_y = gamma*normalized_x + beta
    return normalized_y