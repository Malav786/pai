import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


def train_classifier(model,
    train_ds,
    val_ds,
    device="cpu",
    epochs=30,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    grad_clip=1.0,
    patience=7,
    min_delta=1e-4,):
    """Train a neural network classifier on face recognition data.
    
    Trains the model for the specified number of epochs, performing both training
    and validation after each epoch. Uses Adam optimizer with cross-entropy loss.
    Prints training loss and validation accuracy after each epoch.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_ds (Dataset): Training dataset (PyTorch Dataset).
        val_ds (Dataset): Validation dataset (PyTorch Dataset).
        device (str): Device to train on ('cpu' or 'cuda'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Batch size for training and validation. Default is 64.
    
    Returns:
        nn.Module: The trained model (same instance, modified in-place).
    
    Example:
        >>> from finalproject.models import SmallCNN
        >>> from finalproject.data import NumpyFaceDataset
        >>> model = SmallCNN(in_channels=1, num_classes=7)
        >>> train_ds = NumpyFaceDataset(X_train, y_train, transform=transforms.ToTensor())
        >>> val_ds = NumpyFaceDataset(X_val, y_val, transform=transforms.ToTensor())
        >>> trained_model = train_classifier(model, train_ds, val_ds, epochs=10, batch_size=32)
    """
    model = model.to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, verbose=True
    )

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    bad_epochs = 0

    for e in range(epochs):
        # -------- train --------
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(tr_loader))

        # -------- val --------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += F.cross_entropy(logits, yb).item()
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total)

        scheduler.step(val_loss)

        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            mark = "*"
        else:
            bad_epochs += 1
            mark = ""

        print(
            f"Epoch {e+1}/{epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} {mark}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    return model