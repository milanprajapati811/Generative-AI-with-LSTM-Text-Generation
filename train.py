
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model




def get_callbacks(checkpoint_path: str = "models/best_model.keras"):
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = 5,
            restore_best_weights = True,
            verbose              = 1
        ),
        ModelCheckpoint(
            filepath          = checkpoint_path,
            monitor           = "val_loss",
            save_best_only    = True,
            save_weights_only = False,
            verbose           = 1
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-6,
            verbose  = 1
        ),
    ]
    return callbacks


def train_model(model           : tf.keras.Model,
                X               : np.ndarray,
                y               : np.ndarray,
                epochs          : int   = 30,
                batch_size      : int   = 256,
                val_split       : float = 0.1,
                checkpoint_path : str   = "models/best_model.keras"):
    
    print("\n" + "="*60)
    print("  TRAINING STARTED")
    print(f"  Samples     : {len(X):,}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Val split   : {val_split}")
    print("="*60 + "\n")

    callbacks = get_callbacks(checkpoint_path)

    history = model.fit(
        X, y,
        epochs           = epochs,
        batch_size       = batch_size,
        validation_split = val_split,
        callbacks        = callbacks,
        verbose          = 1
    )

    print("\n[Train] Training complete.")
    return history



def load_best_model(checkpoint_path: str = "models/best_model.keras"):
   
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
        print(f"[Train] Best model loaded from: {checkpoint_path}")
        return model
    else:
        print(f"[Train] WARNING: No checkpoint found at {checkpoint_path}")
        return None



def plot_training_history(history,
                          save_path: str = "outputs/training_history.png"):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Training History", fontsize=16, fontweight="bold")

   
    axes[0].plot(history.history["loss"],     label="Train Loss",      color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Validation Loss", color="#F44336", linewidth=2, linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse Categorical Crossentropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

   
    axes[1].plot(history.history["accuracy"],     label="Train Accuracy",      color="#4CAF50", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#FF9800", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Train] Training history plot saved to: {save_path}")


def compare_architectures(models_dict : dict,
                           X           : np.ndarray,
                           y           : np.ndarray,
                           epochs      : int = 3,
                           batch_size  : int = 256,
                           n_samples   : int = 50_000):
  
    X_sub = X[:n_samples]
    y_sub = y[:n_samples]

    results = {}
    for name, model in models_dict.items():
        print(f"\n[Bonus] Training: {name}")
        h = model.fit(
            X_sub, y_sub,
            epochs           = epochs,
            batch_size       = batch_size,
            validation_split = 0.1,
            verbose          = 0
        )
        vl = h.history["val_loss"][-1]
        va = h.history["val_accuracy"][-1]
        results[name] = {"val_loss": vl, "val_acc": va}
        print(f"  val_loss={vl:.4f}  |  val_accuracy={va:.4f}")

   
    print("\n" + "="*77)
    print("  BONUS: ARCHITECTURE COMPARISON SUMMARY")
    print("="*77)
    print(f"{'Architecture':<52} {'Val Loss':>10} {'Val Acc':>10}")
    print("-"*77)
    for name, r in results.items():
        print(f"{name:<52} {r['val_loss']:>10.4f} {r['val_acc']:>10.4f}")
    print("="*77)

   
    _plot_comparison(results)

    return results


def _plot_comparison(results: dict,
                     save_path: str = "outputs/architecture_comparison.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names     = list(results.keys())
    val_loss  = [results[n]["val_loss"] for n in names]
    val_acc   = [results[n]["val_acc"]  for n in names]
    colors    = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Architecture Comparison (3 Epochs)", fontsize=14, fontweight="bold")

    short_names = ["ShallowLSTM", "DefaultLSTM", "DeepLSTM"]

    axes[0].bar(short_names, val_loss, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Validation Loss (lower = better)")
    axes[0].set_ylabel("Val Loss")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(val_loss):
        axes[0].text(i, v + 0.05, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(short_names, val_acc, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Validation Accuracy (higher = better)")
    axes[1].set_ylabel("Val Accuracy")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(val_acc):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Train] Architecture comparison plot saved to: {save_path}")
