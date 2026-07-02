import torch
import os
import matplotlib.pyplot as plt
import numpy as np

class EarlyStopping:
    def __init__(self, patience: int =5, min_delta: float=0):
        """
        :param patience: How many epochs to wait after last time validation loss improved.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Check if the validation loss improved significantly
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model state
            self.best_model_state_dict = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def save_prediction(
        model,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        epoch: int,
        device: torch.device,
        output_dir: str,
        n_samples: int =3
):
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        X, y = next(iter(dataloader))

        X = X.to(device)
        logits = model(X)

        X = X.to("cpu")
        y = y.to("cpu")
        logits = logits.to("cpu")
        preds = logits.argmax(dim=1)

        # number of samples to show minimum between n_samples and batch_size
        n = min(n_samples, X.shape[0])

        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        for i in range(n): # iterate samples
            # compute loss
            loss = loss_fn(logits[i].unsqueeze(0), y[i].unsqueeze(0)).item()

            # input image
            image = X[i].permute((1,2,0))
            # Reverse the transformation and clip to safety
            img_to_show = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_to_show = np.clip(img_to_show, 0, 1)

            axes[i,0].imshow(img_to_show)
            axes[i,0].set_title("Input")
            axes[i,0].axis("off")

            # ground truth
            axes[i,1].imshow(y[i])
            axes[i,1].set_title("Ground Truth")
            axes[i,1].axis("off")

            # prediction
            axes[i,2].imshow(preds[i])
            axes[i,2].set_title(f"Prediction (Loss: {loss:.4f})")
            axes[i,2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/epoch_{epoch:03d}.png")
        plt.close()
