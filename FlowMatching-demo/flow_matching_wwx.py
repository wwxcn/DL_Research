import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatching(nn.Module):
    def __init__(self, in_dim = 10, hidden_dim = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(in_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, x):
        epi = nn.Random()
        t = nn.
        target = x - epi
        xt = (1-t)*epi + t*x
        v_pre = self.net(xt, t)
        return v_pre, target
    
Class FMLoss(nn.Module):
