import numpy as np
import torch
from torch import nn

__all__ = ['FeatureNormalize']

class FeatureNormalize(nn.Module):
    """
    Output normalize layer
    """
    def __init__(self, p=2, dim=1):
        super().__init__()
        self.p,self.dim = p,dim
    def forward(self, x): return torch.nn.functional.normalize(x, p=self.p)

@torch.no_grad()
def check_encoder_output_if_normalized(encoder, device='cuda:0'):
    state = encoder.training
    encoder.eval()

    x = torch.randn(1,3,224,224, device=device)
    x = encoder(x)
    
    x = (x[0]**2).sum().item()
    assert np.isclose(x, 1), f'encoder output is not normalized: {x}'
    
    print('Output is normalized.')

    if state: encoder.train()
