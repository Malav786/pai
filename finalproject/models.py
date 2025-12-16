import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np


class SmallCNN(nn.Module):
    """Small Convolutional Neural Network for face recognition.

    A lightweight CNN architecture designed for face classification tasks.
    The network consists of two convolutional layers followed by max pooling,
    and fully connected layers with dropout for regularization.

    Architecture:
        - Conv1: 1 channel → 32 channels (3x3 kernel, padding=1)
        - ReLU activation
        - Conv2: 32 channels → 64 channels (3x3 kernel, padding=1)
        - ReLU activation + MaxPool2d (2x2)
        - Flatten
        - Linear: feature_size → 256
        - Dropout (0.3)
        - Linear: 256 → num_classes

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        fc (nn.Sequential): Fully connected layers with flatten, linear, dropout.

    Args:
        in_channels (int): Number of input channels. Default is 1 (grayscale)."""

    def __init__(self, in_channels=1, num_classes=7, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)


class ConvEncoder(nn.Module):
    """Convolutional Encoder for autoencoder architecture.

    Encodes input images into a latent representation for reconstruction.
    Used in model inversion attacks to reconstruct facial images from model outputs.

    Architecture:
        - 3 convolutional layers with stride 2 for downsampling
        - Final feature map: 128 * 6 * 4 = 3072 features
        - Linear layer to latent dimension

    Args:
        latent_dim (int): Dimension of the latent space. Default is 128.
        in_channels (int): Number of input channels. Default is 1 (grayscale).
    """

    def __init__(self, latent_dim, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # For 50x37 images: 50->25->12->6, 37->18->9->4
        # Final feature map: 128 * 6 * 4 = 3072
        self.fc = nn.Linear(128 * 6 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc(self.flatten(h))


class ConvDecoder(nn.Module):
    """Convolutional Decoder for autoencoder architecture.

    Decodes latent representations back into images.
    Used in model inversion attacks to reconstruct facial images from model outputs.

    Architecture:
        - Linear layer to expand latent dimension
        - 3 transpose convolutional layers for upsampling
        - Output matches input dimensions (50x37)

    Args:
        latent_dim (int): Dimension of the latent space. Default is 128.
        out_channels (int): Number of output channels. Default is 1 (grayscale).
    """

    def __init__(self, latent_dim, out_channels):
        super().__init__()
        # For 50x37 images: need to reshape to 128*6*4
        self.fc = nn.Linear(latent_dim, 128 * 6 * 4)
        # Use output_padding to match exact dimensions: 6x4 -> 12x9 -> 25x18 -> 50x37
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=(0, 1)),
            nn.ReLU(),  # 6x4 -> 12x9
            nn.ConvTranspose2d(64, 32, 4, 2, 1, output_padding=(1, 0)),
            nn.ReLU(),  # 12x9 -> 25x18
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1, output_padding=(0, 1)),
            nn.Sigmoid(),  # 25x18 -> 50x37
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 6, 4)
        return self.deconv(h)


def train_autoencoder(
    encoder,
    decoder,
    train_ds,
    val_ds=None,
    device="cpu",
    epochs=30,
    patience=5,
    min_delta=1e-6,
):
    """Train an autoencoder model with early stopping.

    The autoencoder is used for model inversion attacks, not for classification.
    It learns to reconstruct images, which can be used to recover sensitive
    facial information from model outputs.

    Args:
        encoder: Encoder model (ConvEncoder)
        decoder: Decoder model (ConvDecoder)
        train_ds: Training dataset
        val_ds: Optional validation dataset for monitoring (required for early stopping)
        device: Device to train on ('cpu' or 'cuda')
        epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before stopping (default: 5)
        min_delta: Minimum change to qualify as an improvement (default: 1e-6)

    Returns:
        Trained encoder and decoder models (best models based on validation loss)
    """
    encoder, decoder = encoder.to(device), decoder.to(device)
    # Use a lower learning rate to prevent rapid collapse
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=64, shuffle=False) if val_ds is not None else None
    )

    # Early stopping setup
    best_val_loss = float("inf")
    patience_counter = 0
    best_encoder_state = None
    best_decoder_state = None

    for e in range(epochs):
        # Training phase
        encoder.train()
        decoder.train()
        train_loss = 0
        train_batches = 0
        train_psnr = 0
        train_pixel_acc = 0
        pixel_threshold = 0.05  # Threshold for pixel accuracy (5% of pixel value range)
        for batch in train_loader:
            # Handle both cases: dataset returns (img, label) or just img
            if isinstance(batch, tuple):
                xb, _ = batch
            else:
                xb = batch
            xb = xb.to(device)
            z = encoder(xb)
            recon = decoder(z)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

            # Calculate metrics (detach to avoid gradient computation)
            with torch.no_grad():
                # PSNR
                mse = F.mse_loss(recon, xb, reduction="none")
                mse_per_image = mse.view(xb.size(0), -1).mean(dim=1)
                psnr = -10 * torch.log10(mse_per_image + 1e-10)
                train_psnr += psnr.mean().item()

                # Pixel accuracy
                pixel_diff = torch.abs(recon - xb)
                correct_pixels = (pixel_diff < pixel_threshold).float()
                pixel_acc = correct_pixels.view(xb.size(0), -1).mean(dim=1).mean()
                train_pixel_acc += pixel_acc.item()

            train_batches += 1
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        avg_train_psnr = train_psnr / train_batches if train_batches > 0 else 0
        avg_train_pixel_acc = (
            train_pixel_acc / train_batches if train_batches > 0 else 0
        )

        # Validation phase
        if val_loader is not None:
            encoder.eval()
            decoder.eval()
            val_loss = 0
            val_batches = 0
            val_psnr = 0
            val_pixel_acc = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, tuple):
                        xb, _ = batch
                    else:
                        xb = batch
                    xb = xb.to(device)
                    z = encoder(xb)
                    recon = decoder(z)
                    loss = F.mse_loss(recon, xb)
                    val_loss += loss.item()

                    # Calculate PSNR (Peak Signal-to-Noise Ratio)
                    mse = F.mse_loss(recon, xb, reduction="none")
                    mse_per_image = mse.view(xb.size(0), -1).mean(dim=1)
                    psnr = -10 * torch.log10(
                        mse_per_image + 1e-10
                    )  # Add small epsilon to avoid log(0)
                    val_psnr += psnr.mean().item()

                    # Calculate pixel accuracy (percentage of pixels within threshold)
                    pixel_diff = torch.abs(recon - xb)
                    correct_pixels = (pixel_diff < pixel_threshold).float()
                    pixel_acc = correct_pixels.view(xb.size(0), -1).mean(dim=1).mean()
                    val_pixel_acc += pixel_acc.item()

                    val_batches += 1
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            avg_val_psnr = val_psnr / val_batches if val_batches > 0 else 0
            avg_val_pixel_acc = val_pixel_acc / val_batches if val_batches > 0 else 0

            # Early stopping logic
            if avg_val_loss < best_val_loss - min_delta:
                # Improvement found
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model states
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_decoder_state = copy.deepcopy(decoder.state_dict())
                print(
                    f"AE epoch {e+1}/{epochs}: train_loss {avg_train_loss:.6f} (PSNR {avg_train_psnr:.2f}dB, "
                    f"acc {avg_train_pixel_acc*100:.2f}%), val_loss {avg_val_loss:.6f} (PSNR {avg_val_psnr:.2f}dB, "
                    f"acc {avg_val_pixel_acc*100:.2f}%) *"
                )
            else:
                # No improvement
                patience_counter += 1
                print(
                    f"AE epoch {e+1}/{epochs}: train_loss {avg_train_loss:.6f} (PSNR {avg_train_psnr:.2f}dB, "
                    f"acc {avg_train_pixel_acc*100:.2f}%), val_loss {avg_val_loss:.6f} (PSNR {avg_val_psnr:.2f}dB, "
                    f"acc {avg_val_pixel_acc*100:.2f}%) (no improvement, patience: {patience_counter}/{patience})"
                )

                # Early stopping
                if patience_counter >= patience:
                    print(
                        f"\nEarly stopping triggered after {e+1} epochs. Restoring best model (val_loss: {best_val_loss:.6f})"
                    )
                    encoder.load_state_dict(best_encoder_state)
                    decoder.load_state_dict(best_decoder_state)
                    break
        else:
            print(
                f"AE epoch {e+1}/{epochs}: train_loss {avg_train_loss:.6f} (PSNR {avg_train_psnr:.2f}dB, "
                f"acc {avg_train_pixel_acc*100:.2f}%)"
            )

    # Restore best models if early stopping was used
    if val_loader is not None and best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)

    return encoder, decoder


def train_classifier(
    model,
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
    min_delta=1e-4,
):
    """Train a neural network classifier on face recognition data.

    Trains the model for the specified number of epochs, performing both training
    and validation after each epoch. Uses Adam optimizer with cross-entropy loss.
    Prints training loss and validation accuracy after each epoch.

    Args:
        model (nn.Module): The neural network model to train.
        train_ds (Dataset): Training dataset (PyTorch Dataset).
        val_ds (Dataset): Validation dataset (PyTorch Dataset).
        device (str): Device to train on ('cpu' or 'cuda'). Default is 'cpu'.
        epochs (int): Number of training epochs. Default is 30.
        batch_size (int): Batch size for training and validation. Default is 64.
        lr (float): Learning rate. Default is 3e-4.
        weight_decay (float): Weight decay for optimizer. Default is 1e-4.
        label_smoothing (float): Label smoothing factor. Default is 0.05.
        grad_clip (float): Gradient clipping value. Default is 1.0.
        patience (int): Number of epochs to wait for improvement before stopping. Default is 7.
        min_delta (float): Minimum change to qualify as an improvement. Default is 1e-4.

    Returns:
        nn.Module: The trained model (same instance, modified in-place).

    Example:
        >>> from finalproject.models import SmallCNN, train_classifier
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


def extract_features(encoder, dataset, device="cpu"):
    """
    Extract features from images using the trained encoder.

    Args:
        encoder: Trained encoder model
        dataset: Dataset to extract features from
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        If dataset has labels: (features, labels) as numpy arrays
        Otherwise: features as numpy array
    """
    encoder.eval()
    features = []
    labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            # Handle batch - DataLoader returns tuple (images, labels) when dataset has labels
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                xb, yb = batch
            else:
                xb = batch
                yb = None

            # Ensure xb is a tensor
            if not isinstance(xb, torch.Tensor):
                xb = torch.tensor(xb) if isinstance(xb, (list, np.ndarray)) else xb

            xb = xb.to(device)
            z = encoder(xb)  # Extract latent features
            features.append(z.cpu().numpy())

            if yb is not None:
                # Convert labels to numpy if needed
                if isinstance(yb, torch.Tensor):
                    labels.append(yb.cpu().numpy())
                else:
                    labels.append(np.array(yb))

    features = np.concatenate(features, axis=0)
    if labels:
        labels = np.concatenate(labels, axis=0)
        return features, labels
    return features
