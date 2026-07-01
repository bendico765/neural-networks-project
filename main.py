import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import optuna
import camvid
import engine
import argparse
import os
import utils
import albumentations as A
from datetime import datetime
import visualization

import metrics
import unet
import fcn
import segnet

# picking device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")

# path to the root folder of the data
parser = argparse.ArgumentParser(description="")
parser.add_argument("data_root_filepath", help="Path to the project data root directory.")
parser.add_argument("--model", type=str, choices=["unet", "segnet", "fcn"], default="unet", help="The model to use.")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for hyperparameter optimization")
parser.add_argument("--batch-size", type=int, default=20, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for hyperparameter optimization and to re-train the final model")
parser.add_argument("--patience", type=int, default=5, help="Number of epochs patience for early stopping")
parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum delta value for early stopping")
parser.add_argument("--random-state", type=int, default=None, help="Random state used for loading up data")
parser.add_argument(
    "--enable-optimization",
    action="store_true",
    help="Whether to optimize the model, or use the command-line provided arguments. If this flag is not specified, just use the command line arguments to train a model (on train and validation sets) and evaluate on the test set.")
parser.add_argument("--test", action="store_true", help="Whether or not to retrain the best model on the test and validation set, and run it on the test")
args = parser.parse_args()

# Command-line parameters
data_root_filepath = args.data_root_filepath
model_type = args.model
learning_rate = args.lr
n_trials = args.n_trials
batch_size = args.batch_size
epochs = args.epochs
patience = args.patience
min_delta = args.min_delta
random_state = args.random_state
enable_optimization = args.enable_optimization

# save configuration for the current run
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(f"{data_root_filepath}/runs"):
    os.makedirs(f"{data_root_filepath}/runs")

if not os.path.exists(f"{data_root_filepath}/runs/{run_name}"):
    os.makedirs(f"{data_root_filepath}/runs/{run_name}")


# defining transforms to augment data
"""
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=270, p=1.0)
])
"""
transforms = A.Compose([
    A.RandomResizedCrop(size=(224, 224)),
    A.HorizontalFlip(p=0.5),
    A.Normalize()
])

# define loss
loss_fn = nn.CrossEntropyLoss(ignore_index=11)

### LOADING DATA
train_data = camvid.CAMVID_Dataset(f"{data_root_filepath}/train", f"{data_root_filepath}/train_labels", labels_filepath=f"{data_root_filepath}/class_palette.csv", transform=transforms)
validation_data = camvid.CAMVID_Dataset(f"{data_root_filepath}/val", f"{data_root_filepath}/val_labels", labels_filepath=f"{data_root_filepath}/class_palette.csv", transform=transforms)
test_data = camvid.CAMVID_Dataset(f"{data_root_filepath}/test", f"{data_root_filepath}/test_labels", labels_filepath=f"{data_root_filepath}/class_palette.csv")

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
validation_dataloader = DataLoader(
    validation_data,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

print(f"\nData samples------------", flush=True)
print(f"Training:{len(train_data)}", flush=True)
print(f"Validation:{len(validation_data)}", flush=True)
print(f"Test:{len(test_data)}", flush=True)

### HYPERPARAMETER OPTIMIZATION
if enable_optimization:
    print("\nOptimizing hyperparameters--------", flush=True)

    # create directory to save the trials for the search
    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/trials"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/trials")

    # use optuna for hyperparameters optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(
        engine.Objective(
            f"{data_root_filepath}/runs/{run_name}/trials",
            model_type,
            train_dataloader,
            validation_dataloader,
            loss_fn,
            epochs,
            patience,
            min_delta,
            device
        ),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    print(f"Best hyperparameters: {study.best_params}", flush=True)
    print(f"User attrs: {study.best_trial.user_attrs}", flush=True)

    # save up the best hyperparameters
    learning_rate = study.best_params["lr"]
    batch_size = study.best_params["batch-size"]
    epochs = best_trial.user_attrs["epochs"]
else:
    # just use train and validation set using the command line provided arguments
    print("\nTraining on train set and evaluating on validation set--------", flush=True)

    # creating model
    if model_type == "unet":
        model = unet.UNet(in_channels=3, out_channels=11)
    elif model_type == "segnet":
        model = segnet.SegNet(in_channels=3, out_channels=11)
    else:
        model= fcn.FCN(in_channels=3, out_channels=11)

    model.to(device)

    # defining optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # initializing early stopping
    early_stopper = utils.EarlyStopping(patience=patience, min_delta=min_delta)

    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/model"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/model")

    ### TRAIN THE MODEL
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}\n------------", flush=True)

        # training
        train_loss = engine.train_loop(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            batch_size,
            device
        )
        train_losses.append(train_loss)

        # validation
        val_loss = engine.test_loop(
            validation_dataloader,
            model,
            loss_fn,
            device
        )
        val_losses.append(val_loss)

        # each few epoch save some predicted samples
        """
        if epoch % 4 == 0:
            utils.save_prediction(
                model,
                validation_dataloader,
                loss_fn,
                epoch,
                device,
                f"{data_root_filepath}/runs/{run_name}/model/prediction_samples"
            )
        """
        # logging
        print(f"\nAvg. train loss={train_loss:.6f}\nAvg. val loss={val_loss:.6f}\n", flush=True)

        # saving model checkpoints
        if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/model/checkpoints"):
            os.makedirs(f"{data_root_filepath}/runs/{run_name}/model/checkpoints")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_loss": train_loss
        },
            f"{data_root_filepath}/runs/{run_name}/model/checkpoints/checkpoint_{epoch}.pth"
        )

        # checking early stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered", flush=True)
            epochs = epoch + 1

            # save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": early_stopper.best_model_state_dict,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "val_loss": early_stopper.best_loss
            },
            f"{data_root_filepath}/runs/{run_name}/model/checkpoints/best_model_checkpoint.pth"
            )
            break

    ### LOGGING ###
    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/model/logs"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/model/logs")

    # saving up the loss history
    history = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    history.to_csv(f"{data_root_filepath}/runs/{run_name}/model/logs/loss_history.csv", index=False)

    # plotting loss history
    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/model/figs"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/model/figs")
    visualization.plot_losses(
        history,
        filepath=f"{data_root_filepath}/runs/{run_name}/model/figs/loss_history.png"
    )

### RETRAIN THE BEST MODEL ON THE WHOLE DATASET AND TEST IT
if args.test:
    trainval_dataloader = DataLoader(
        dataset=torch.utils.data.ConcatDataset([train_data, validation_data]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print("\nTRAINING THE FINAL MODEL AND EVALUATING ON THE TEST SET", flush=True)
    print(f"\nData samples------------", flush=True)
    print(f"Training:{len(train_data) + len(validation_data)}", flush=True)
    print(f"Test:{len(test_data)}", flush=True)

    # creating model
    if model_type == "unet":
        model = unet.UNet(in_channels=3, out_channels=11)
    elif model_type == "segnet":
        model = segnet.SegNet(in_channels=3, out_channels=11)
    else:
        model = fcn.FCN(in_channels=3, out_channels=11)
    model.to(device)

    # defining optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # define loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=11)

    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/final_model"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/final_model")

    ### TRAIN THE MODEL
    train_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}\n------------", flush=True)

        # training
        train_loss = engine.train_loop(trainval_dataloader, model, loss_fn, optimizer, batch_size, device)
        train_losses.append(train_loss)

        # each few epoch save some predicted samples
        """
        if epoch % 4 == 0:
            utils.save_prediction(
                model,
                trainval_dataloader,
                loss_fn,
                epoch,
                device,
                f"{data_root_filepath}/runs/{run_name}/final_model/prediction_samples"
            )
        """
        # logging
        print(f"\nAvg. train loss={train_loss:.6f}\n", flush=True)

        # saving model checkpoints
        if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/final_model/checkpoints"):
            os.makedirs(f"{data_root_filepath}/runs/{run_name}/final_model/checkpoints")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_loss": train_loss
        },
            f"{data_root_filepath}/runs/{run_name}/final_model/checkpoints/checkpoint_{epoch}.pth"
        )

    # Make inference on the test set
    print("\nINFERENCE ON THE TEST SET", flush=True)
    test_loss = engine.test_loop(
        test_dataloader,
        model,
        loss_fn,
        device
    )
    print(f"Test loss={test_loss:.6f}", flush=True)

    ### LOGGING ###
    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/final_model/logs"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/final_model/logs")

    # saving up the loss history
    history = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "train_loss": train_losses
    })
    history.to_csv(f"{data_root_filepath}/runs/{run_name}/final_model/logs/loss_history.csv", index=False)