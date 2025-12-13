import torch, torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 

def train_classifier(model, train_ds, val_ds, device='cpu', epochs=20, batch_size=64):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    for e in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
    # validation
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
    print(f"Epoch {e}, train_loss {total_loss/len(tr_loader):.3f}, val_acc {acc:.3f}")
    return model
