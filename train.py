import torch
import torch.nn as nn
from livelossplot import PlotLosses

def train_autoencoder(model, 
                      dataloader, 
                      criterion=torch.nn.MSELoss(reduction='sum'), 
                      optimizer=None, 
                      device="cpu", 
                      num_epoch=50):
    
    if optimizer == None:
        print("Using Adam optimizer with default learning rate")
        torch.optim.Adam(model.parameters())
        
    liveloss = PlotLosses()
    loss_history = {}
    model.train()
    for epoch in range(num_epoch+1):
        logs = {}
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, inputs.float())
            if epoch > 0:
                optimizer.zero_grad()
                loss.backward()                
                optimizer.step()            
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader.dataset)
        logs['loss'] = epoch_loss   
        loss_history.setdefault("train", []).append(epoch_loss)
    
        liveloss.update(logs)
        liveloss.send()
    return loss_history, liveloss