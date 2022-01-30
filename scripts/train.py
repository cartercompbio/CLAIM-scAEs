import torch
import torch.nn as nn
from livelossplot import PlotLosses

def train_autoencoder(model, 
                      dataloader, 
                      criterion=torch.nn.MSELoss(reduction='sum'), 
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
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, inputs.float())
            if epoch > 0:
                optimizer.zero_grad()
                loss.backward()                
                optimizer.step()            
            running_loss += loss.item()

        # Save loss scaled for single sample
        epoch_loss = running_loss / len(dataloader.dataset)
        logs['loss'] = epoch_loss   
        loss_history.setdefault("train", []).append(epoch_loss)
    
        # Update loss graph
        liveloss.update(logs)
        
        if epoch % plot_frequency == 0:
            liveloss.send()
        
    return loss_history, liveloss