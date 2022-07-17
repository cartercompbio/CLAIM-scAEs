import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from livelossplot import PlotLosses

# Define a few custom activations for the mean and dispersion layers of the model
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)


class VanillaAE(nn.Module):
    def __init__(self, input_size, hidden=2):
        super(VanillaAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, hidden)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    
# Class for the DCA neural network
class DCA(nn.Module):
    def __init__(self, input_size, hidden=10):
        super(DCA, self).__init__()
        
        self.input_size = input_size
        
        # Encode the input
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, hidden)
        )
        
        # Decode the input
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
        )
        
        # Get the parameters for each gene
        self.pi = nn.Linear(512, self.input_size)
        self.mean = nn.Linear(512, self.input_size)
        self.disp = nn.Linear(512, self.input_size)
    
    def forward(self, x):
        x = self.decoder(self.encoder(x))
        pi = torch.sigmoid(self.pi(x))
        mean = MeanAct(self.mean(x))
        disp = DispAct(self.disp(x))
        return mean, disp, pi