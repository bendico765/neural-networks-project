import torch
from torch.utils.data import DataLoader

import unet
import fcn
from segnet import SegNet
import os
import pandas as pd
import visualization
import utils

def train_loop(
        dataloader: torch.utils.data.DataLoader,
        model,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        device: torch.device = torch.device("cpu")
    ):
    model.train() # set the model to training mode
    
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device) # (B, 1, W, H)
        y = y.to(device) # (B, 2, W, H) 
        
        # compute prediction and loss
        logits = model(X)

        loss = loss_fn(logits, y)

        # backpropagation
        loss.backward() # backpropagate the prediction loss
        optimizer.step() # adjust the parameters by the gradients collected in the backward pass
        optimizer.zero_grad() # reset the gradients of model parameters

        loss, current = loss.item(), batch * batch_size + len(X)
        total_loss += loss
        
        print(f"loss: {loss:>6f}  [{current:>5d}/{size:>5d}]", flush=True)

    return total_loss/len(dataloader)

def test_loop(
        dataloader: torch.utils.data.DataLoader,
        model,
        loss_fn,
        device: torch.device = torch.device("cpu")
):
    model.eval() # set model to evaluation mode
    
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)

            test_loss += loss_fn(logits, y).item()

    return test_loss / len(dataloader) # return average on the batches

class Objective:
    def __init__(self, 
                 trial_folder_filepath: str,
                 model_type: str,
                 train_dataloader: DataLoader, 
                 validation_dataloader: DataLoader,
                 loss_fn: torch.nn.Module,
                 epochs: int,
                 patience: int,
                 min_delta: float,
                 device: torch.device):
        self.trial_folder_filepath = trial_folder_filepath
        self.model_type = model_type
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.device = device
        
    def __call__(self, trial):
        # create folder to save trial information
        if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}"):
            os.makedirs(f"{self.trial_folder_filepath}/{trial.number}")

        # setting hyperparameters range of values
        learning_rate = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 1e-2, 9e-1, log=True)
        batch_size = trial.suggest_categorical("batch-size", [2, 4, 8, 16])
        
        # creating model
        if self.model_type == "unet":
            model = unet.UNet(in_channels=3, out_channels=11)
        elif self.model_type == "segnet":
            model = SegNet(in_channels=3, out_channels=11)
        else:
            model = fcn.FCN(in_channels=3, out_channels=11)
        model.to(self.device)
        
        # defining optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
    
        # defining loss function
        loss_fn = self.loss_fn
    
        # initializing early stopping
        early_stopper = utils.EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        
        ### Training and evaluating the model
        train_losses = []
        val_losses = []
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}\n------------", flush=True)
            
            # training
            train_loss = train_loop(
                self.train_dataloader, 
                model, 
                loss_fn, 
                optimizer, 
                batch_size,
                self.device
            )
            train_losses.append(train_loss)
            
            # validation
            val_loss = test_loop(self.validation_dataloader, model, loss_fn, self.device)
            val_losses.append(val_loss)

            print(f"\nAvg. train loss={train_loss:.6f}\nAvg. val loss={val_loss:.6f}\n", flush=True)

            # checking early stopping
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered", flush=True)
                self.epochs = epoch + 1

                if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}/checkpoints"):
                    os.makedirs(f"{self.trial_folder_filepath}/{trial.number}/checkpoints")

                # save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": early_stopper.best_model_state_dict,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": self.epochs,
                    "val_loss": early_stopper.best_loss
                },
                    f"{self.trial_folder_filepath}/{trial.number}/checkpoints/best_model_checkpoint.pth"
                )
                break

            if epoch == self.epochs-1: # saving model checkpoints
                if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}/checkpoints"):
                    os.makedirs(f"{self.trial_folder_filepath}/{trial.number}/checkpoints")

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": self.epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                },
                    f"{self.trial_folder_filepath}/{trial.number}/checkpoints/checkpoint_{epoch}.pth"
                )

        # saving the number of epochs performed (useful in case of early stopping)
        trial.set_user_attr("epochs", self.epochs)

        ### LOGGING ###
        if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}/logs"):
            os.makedirs(f"{self.trial_folder_filepath}/{trial.number}/logs")

        # saving up the loss history
        history = pd.DataFrame({
            "epoch": range(1, self.epochs+1),
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        history.to_csv(f"{self.trial_folder_filepath}/{trial.number}/logs/loss_history.csv", index=False)

        # plotting loss history
        if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}/figs"):
            os.makedirs(f"{self.trial_folder_filepath}/{trial.number}/figs")
        visualization.plot_losses(history, filepath=f"{self.trial_folder_filepath}/{trial.number}/figs/loss_history.png")
        
        return val_loss
