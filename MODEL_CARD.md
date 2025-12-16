# Model Card

## Model Overview

This project implements two primary models for face recognition and model inversion attack research:

1. **SmallCNN**: A supervised convolutional neural network classifier for face recognition
2. **Autoencoder (ConvEncoder + ConvDecoder)**: An encoder-decoder architecture for image reconstruction used in model inversion attacks

---

## Model 1: SmallCNN Classifier

### Model Details

- **Architecture**: Convolutional Neural Network
- **Input**: Grayscale facial images (50 × 37 pixels, 1 channel)
- **Output**: 7-class classification logits (7 identity classes)
- **Framework**: PyTorch

### Architecture Specifications

- **Convolutional Layers**:
  - Conv1: 1 → 32 channels (3×3 kernel, padding=1) + BatchNorm + ReLU
  - Conv2: 32 → 64 channels (3×3 kernel, padding=1) + BatchNorm + ReLU
  - MaxPool2d (2×2)
  - Conv3: 64 → 128 channels (3×3 kernel, padding=1) + BatchNorm + ReLU
  - AdaptiveAvgPool2d (4×4)

- **Fully Connected Layers**:
  - Linear: 128 × 4 × 4 → 256
  - ReLU + Dropout (0.3)
  - Linear: 256 → 7 (num_classes)

### Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Weight Decay**: 1e-4
- **Label Smoothing**: 0.05
- **Gradient Clipping**: 1.0
- **Batch Size**: 64
- **Epochs**: 30
- **Early Stopping**: Patience of 7 epochs, min_delta=1e-4
- **Loss Function**: Cross-entropy with label smoothing

### Final Metrics

- **Test Accuracy**: **90.31%** (0.9031)
- **Best Validation Accuracy**: 93.80% (achieved at epoch 28)
- **Final Validation Loss**: 0.2243
- **Final Training Loss**: 0.3827

### Evaluation Details

- **Test Set Size**: 258 samples
- **Evaluation Metric**: Classification accuracy
- **Per-Class Performance**: Available in classification report (precision, recall, F1-score for each of 7 identity classes)

### Domain-Specific Metrics

- **Real Image Classification Accuracy**: 93.33% (on selected real test images)
- **Reconstructed Image Classification Accuracy**: 100.00% (on attack-reconstructed images)
- **Mixed Set Accuracy**: 93.94% (real + reconstructed images combined)

---

## Model 2: Autoencoder (ConvEncoder + ConvDecoder)

### Model Details

- **Purpose**: Image reconstruction for model inversion attacks
- **Architecture**: Encoder-Decoder (Convolutional)
- **Input**: Grayscale facial images (50 × 37 pixels, 1 channel)
- **Output**: Reconstructed grayscale images (50 × 37 pixels, 1 channel)
- **Latent Dimension**: 64 (configurable)

### Architecture Specifications

**Encoder (ConvEncoder)**:
- Conv1: 1 → 32 channels (4×4 kernel, stride=2, padding=1) + ReLU
- Conv2: 32 → 64 channels (4×4 kernel, stride=2, padding=1) + ReLU
- Conv3: 64 → 128 channels (4×4 kernel, stride=2, padding=1) + ReLU
- Flatten + Linear: 128 × 6 × 4 → latent_dim

**Decoder (ConvDecoder)**:
- Linear: latent_dim → 128 × 6 × 4
- Reshape to (128, 6, 4)
- ConvTranspose1: 128 → 64 channels (4×4 kernel, stride=2, padding=1) + ReLU
- ConvTranspose2: 64 → 32 channels (4×4 kernel, stride=2, padding=1) + ReLU
- ConvTranspose3: 32 → 1 channel (4×4 kernel, stride=2, padding=1) + Sigmoid

### Training Details

- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)
- **Early Stopping**: Patience of 5 epochs, min_delta=1e-6
- **Loss Function**: Mean Squared Error (MSE)

### Final Metrics

- **Classification Accuracy (with SVM on extracted features)**: **71.32%** (0.7132)
  - Note: The autoencoder is not intended as a competitive classifier; it serves as a reconstruction tool for model inversion attacks.

### Reconstruction Quality Metrics

When used in model inversion attacks:
- **Average SSIM (Structural Similarity Index)**: 0.1197 (range: 0.0427-0.2224)
- **Average PSNR (Peak Signal-to-Noise Ratio)**: 14.83 dB (range: 12.41-17.32 dB)
- **Best Reconstruction**: 0.2224 SSIM and 17.32 dB PSNR (George W Bush class)

### Domain-Specific Metrics

- **Attack Success Rate**: 100% (all attacks achieved >95% target class confidence)
- **Average Attack Confidence**: 98.53% (0.9853)
- **Queries Required**: ~5,050 queries per successful attack (101 iterations × 50 population size)

---

## Training and Evaluation Methodology

### Data Split

- **Training Set**: 901 samples
- **Validation Set**: 129 samples
- **Test Set**: 258 samples
- **Total Dataset**: 1,288 grayscale facial images (50×37 pixels)
- **Classes**: 7 identity classes (people with ≥70 images each)

### Training Procedure

1. **SmallCNN**:
   - Trained for 30 epochs with early stopping
   - Best model selected based on validation accuracy
   - Model checkpointing implemented to restore best weights
   - Gradient clipping applied to prevent exploding gradients

2. **Autoencoder**:
   - Trained for 50 epochs with early stopping
   - Best model selected based on validation loss
   - PSNR and pixel accuracy tracked during training
   - Model checkpointing implemented

### Evaluation Procedure

- **SmallCNN**: Evaluated on held-out test set (258 samples) using classification accuracy
- **Autoencoder**: Evaluated indirectly through:
  1. Classification accuracy when features are extracted and used with SVM classifier
  2. Reconstruction quality metrics (SSIM, PSNR) when used in model inversion attacks

### Reproducibility

- **Random Seed**: 42 (set using `set_seed()` function)
- **Deterministic Operations**: Enabled for PyTorch (where possible)
- **Device**: CPU (configurable to CUDA if available)

---

## Known Limitations

### SmallCNN Classifier

1. **Dataset Size**: Trained on a small subset of LFW (7 people, 1,288 images), limiting generalization to broader face recognition tasks
2. **Image Resolution**: Low resolution (50×37 pixels) may not capture fine facial features
3. **Class Imbalance**: Some classes have more samples than others, potentially affecting per-class performance
4. **Overfitting Risk**: With limited data, the model may overfit to training patterns
5. **Vulnerability to Attacks**: The model is vulnerable to model inversion attacks, achieving 98.53% average attack confidence with ~5,050 queries

### Autoencoder

1. **Reconstruction Quality**: Low SSIM (0.1197 average) and PSNR (14.83 dB average) indicate imperfect pixel-level reconstruction
2. **Classification Performance**: 71.32% accuracy when used as a feature extractor is significantly lower than the supervised CNN (90.31%)
3. **Purpose Limitation**: Designed for reconstruction in attacks, not for competitive classification
4. **Latent Space**: Fixed latent dimension (64) may not be optimal for all identity classes
5. **Attack Dependency**: Performance is evaluated in the context of model inversion attacks, not standalone image reconstruction

### General Limitations

1. **Limited Identity Classes**: Only 7 identity classes evaluated, not representative of real-world face recognition systems with thousands of identities
2. **Controlled Environment**: Experiments conducted in controlled research setting, may not reflect real-world deployment conditions
3. **Defense Mechanisms**: While defenses are implemented, no single defense provides complete protection against model inversion attacks
4. **Computational Resources**: Training and evaluation performed on limited computational resources, may not scale to production systems

---

## Intended Use

### SmallCNN

- **Primary Use**: Face recognition/classification on the 7 identity classes from the filtered LFW dataset
- **Research Context**: Serves as the "victim model" in model inversion attack demonstrations
- **Not Intended For**: Production face recognition systems, large-scale identity verification, or security-critical applications without additional defenses

### Autoencoder

- **Primary Use**: Image reconstruction component in model inversion attack framework
- **Research Context**: Demonstrates how private training data can be reconstructed from model outputs
- **Not Intended For**: Standalone image reconstruction, competitive face classification, or production image generation systems

---

## Model Files

- **Model Definitions**: `finalproject/models.py`
  - `SmallCNN`: Classifier architecture
  - `ConvEncoder`: Encoder architecture
  - `ConvDecoder`: Decoder architecture
  - `train_classifier()`: Training function for classifier
  - `train_autoencoder()`: Training function for autoencoder

- **Training Scripts**: Training performed in `notebooks/finalproject.ipynb`
