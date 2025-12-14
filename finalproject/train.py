import torch, torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 


def train_classifier(model, train_ds, val_ds, device='cpu', epochs=20, batch_size=64):
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
    opt = optim.Adam(model.parameters(), lr=1e-3)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    for e in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct/total
        print(f"Epoch {e+1}/{epochs}, train_loss {total_loss/len(tr_loader):.3f}, val_acc {acc:.3f}")
    return model