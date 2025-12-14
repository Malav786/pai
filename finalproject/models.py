import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super().__init__()
        c = 32
        self.conv1 = nn.Conv2d(in_channels, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c*2, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Sequential(nn.Flatten(), 
                                nn.Linear((c*2)*(int(62*0.5)**2), 256), # adjust size to image dims nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, num_classes)
                               ) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.fc(x)
        return x

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
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # For 50x37 images: 50->25->12->6, 37->18->9->4
        # Final feature map: 128 * 6 * 4 = 3072
        self.fc = nn.Linear(128*6*4, latent_dim)
    
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
        self.fc = nn.Linear(latent_dim, 128*6*4)
        # Use output_padding to match exact dimensions: 6x4 -> 12x9 -> 25x18 -> 50x37
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, output_padding=(0, 1)), nn.ReLU(),  # 6x4 -> 12x9
            nn.ConvTranspose2d(64, 32, 4, 2, 1, output_padding=(1, 0)), nn.ReLU(),   # 12x9 -> 25x18
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1, output_padding=(0, 1)), nn.Sigmoid()  # 25x18 -> 50x37
        )
    
    def forward(self, z):
        h = self.fc(z).view(-1, 128, 6, 4)
        return self.deconv(h)

def train_autoencoder(encoder, decoder, train_ds, val_ds=None, device='cpu', epochs=30, 
                     patience=5, min_delta=1e-6):
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
    opt = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-4)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False) if val_ds is not None else None
    
    # Early stopping setup
    best_val_loss = float('inf')
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
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
            
            # Calculate metrics (detach to avoid gradient computation)
            with torch.no_grad():
                # PSNR
                mse = F.mse_loss(recon, xb, reduction='none')
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
        avg_train_pixel_acc = train_pixel_acc / train_batches if train_batches > 0 else 0
        
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
                    mse = F.mse_loss(recon, xb, reduction='none')
                    mse_per_image = mse.view(xb.size(0), -1).mean(dim=1)
                    psnr = -10 * torch.log10(mse_per_image + 1e-10)  # Add small epsilon to avoid log(0)
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
                print(f"AE epoch {e+1}/{epochs}: train_loss {avg_train_loss:.6f} (PSNR {avg_train_psnr:.2f}dB, "
                      f"acc {avg_train_pixel_acc*100:.2f}%), val_loss {avg_val_loss:.6f} (PSNR {avg_val_psnr:.2f}dB, "
                      f"acc {avg_val_pixel_acc*100:.2f}%) *")
            else:
                # No improvement
                patience_counter += 1
                print(f"AE epoch {e+1}/{epochs}: train_loss {avg_train_loss:.6f} (PSNR {avg_train_psnr:.2f}dB, "
                      f"acc {avg_train_pixel_acc*100:.2f}%), val_loss {avg_val_loss:.6f} (PSNR {avg_val_psnr:.2f}dB, "
                      f"acc {avg_val_pixel_acc*100:.2f}%) (no improvement, patience: {patience_counter}/{patience})")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {e+1} epochs. Restoring best model (val_loss: {best_val_loss:.6f})")
                    encoder.load_state_dict(best_encoder_state)
                    decoder.load_state_dict(best_decoder_state)
                    break
        else:
            print(f"AE epoch {e+1}/{epochs}: train_loss {avg_train_loss:.6f} (PSNR {avg_train_psnr:.2f}dB, "
                  f"acc {avg_train_pixel_acc*100:.2f}%)")
    
    # Restore best models if early stopping was used
    if val_loader is not None and best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)
    
    return encoder, decoder
