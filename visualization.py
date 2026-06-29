import matplotlib.pyplot as plt
import pandas as pd

def plot_losses(df: pd.DataFrame, figsize: tuple[int, int] =(8, 5), validation=True, filepath=None):
    """
    Plot training and validation loss curves
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        df.index,
        df["train_loss"],
        label="Training Loss",
        marker="o"
    )

    if validation:
        ax.plot(
            df.index,
            df["val_loss"],
            label="Validation Loss",
            marker="o"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Model Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()