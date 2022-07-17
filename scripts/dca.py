import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from livelossplot import PlotLosses


# Define a few custom activations for the mean and dispersion layers of the model
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)

    
# Eventually incorporate this to build custom autoencoders within the DCA framework
def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


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

    
def test_config(model,
                dataloader,
                criterion=ZINBLoss,
                optimizer=None,
                device="cpu"):
    
    # Get a batch from the dataset
    x_raw = dataloader.dataset[:5][0].to(device)
    x_norm = dataloader.dataset[:5][1].to(device)
    s = dataloader.dataset[:5][2].to(device)
    
    # Calculate the loss and parameters of the model
    out = model(x_norm)
    mean, disp, pi = out[0], out[1], out[2]
    pre_params = list(model.parameters())[0].clone()
    loss = criterion(x_raw, mean, disp, pi, s)
    print("Loss before test optimization:", loss.item())
    
    # Take a step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check the parameters and loss
    out = model(x_norm)
    mean, disp, pi = out[0], out[1], out[2]
    gradients = list(model.parameters())[0].grad
    if (gradients != None) and (not torch.all(gradients == 0)):
        print("There are gradients and they are not 0")
    post_params = list(model.parameters())[0].clone()
    if not torch.equal(pre_params.data, post_params.data):
        print("Parameters have been updated")
    loss = criterion(x_raw, mean, disp, pi, s)
    print("Loss after test optimization:", loss.item())
    print("Did it go down?")
    
def train(model, 
          dataloader, 
          criterion=ZINBLoss,
          optimizer=None, 
          device="cpu", 
          num_epoch=50,
          plot_frequency=10):
    
    if optimizer == None:
        print("Using Adam optimizer with default learning rate")
        torch.optim.Adam(model.parameters())
        
    # Variables to save training history
    liveloss = PlotLosses()
    loss_history = {}
    
    # Set model to training mode
    model.train()
    
    # Training loop for num_epochs
    for epoch in range(num_epoch+1):
        logs = {}
        running_loss = 0.0
        
        # Iterate through each batch
        for raw_inputs, norm_inputs, sfs in dataloader:
            raw_inputs = raw_inputs.to(device)
            norm_inputs = norm_inputs.to(device)
            sfs = sfs.to(device)
            outputs = model(norm_inputs.float())
            loss = criterion(raw_inputs, outputs[0], outputs[1], outputs[2], sfs)
            if epoch > 0:
                optimizer.zero_grad()
                loss.backward()                
                optimizer.step()            
            running_loss += loss.item()

        # Save loss scaled for single sample
        epoch_loss = running_loss
        logs['loss'] = epoch_loss   
        loss_history.setdefault("train", []).append(epoch_loss)
    
        # Update loss graph
        liveloss.update(logs)
        
        if epoch % plot_frequency == 0:
            liveloss.send()
        
    return loss_history, liveloss

