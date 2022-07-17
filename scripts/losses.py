import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Zero-inflated negative binomial NLL to minimize during training
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        
        # Rescaling the autoencoder counts, keeps optimization blind to library size differences
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        # Calculating negative binomial log likelihood
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2
        nb_case = nb_final - torch.log(1.0-pi+eps)
        
        # Handle when the count is zero or effectively 0
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        # Regularization of dropout probability
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        # Take the average
        result = torch.mean(result)
        return result