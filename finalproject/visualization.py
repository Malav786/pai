import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def visualize_classifier_predictions(model, dataset, target_names, device='cpu', num_images=8):
    """
    Visualizes a batch of test images with their true labels and model predictions.

    Args:
        model (torch.nn.Module): The trained classification model.
        dataset (torch.utils.data.Dataset): The dataset to draw images from (e.g., test_ds).
        target_names (list): A list of class names.
        device (str): The device to run the model on ('cpu' or 'cuda').
        num_images (int): The number of images to display (max 8 per row).
    """
    model.eval()
    test_loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    with torch.no_grad():
        xb, yb = next(iter(test_loader))
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        yb_np = yb.numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
        for i in range(min(num_images, len(xb))):
            img = xb[i, 0].cpu().numpy()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'True: {target_names[yb_np[i]]}', fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(img, cmap='gray')
            color = 'green' if preds[i] == yb_np[i] else 'red'
            axes[1, i].set_title(f'Pred: {target_names[preds[i]]}',
                                fontsize=8, color=color)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

        batch_acc = (preds == yb_np).mean()
        print(f"Sample batch accuracy: {batch_acc:.2%}")


def visualize_attack_results(attack_results, nes_pop, target_names):
    """
    Visualizes reconstructed images from a model inversion attack and their optimization history.

    Args:
        attack_results (dict): Dictionary containing attack results for each target class.
        nes_pop (int): Population size used in NES attack.
        target_names (list): A list of class names.
    """
    n_classes = len(attack_results)
    fig, axes = plt.subplots(2, n_classes, figsize=(5 * n_classes, 10))

    for idx, (class_idx, result) in enumerate(attack_results.items()):
        class_name = target_names[class_idx]
        best_image = result['best_image']
        history = result['history']
        final_confidence = result['final_confidence']

        axes[0, idx].imshow(best_image, cmap='gray')
        axes[0, idx].set_title(f'{class_name}\nConfidence: {final_confidence:.3f}',
                               fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')

        axes[1, idx].plot(history, linewidth=2)
        axes[1, idx].set_xlabel('Iteration', fontsize=10)
        axes[1, idx].set_ylabel('Target Class Probability', fontsize=10)
        axes[1, idx].set_title('Optimization Progress', fontsize=11)
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_ylim([0, 1.0])

    plt.suptitle('Model Inversion Attack Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("ATTACK SUMMARY")
    print("=" * 60)
    for class_idx, result in attack_results.items():
        print(f"{target_names[class_idx]:20s} | Confidence: {result['final_confidence']:.4f} | "
              f"Queries: {len(result['history']) * nes_pop}")
    print("=" * 60)


def visualize_real_vs_reconstructed(real_images, real_labels, real_predictions, real_confidences,
                                    reconstructed_images, reconstructed_labels, reconstructed_predictions, reconstructed_confidences,
                                    target_names, target_class_indices):
    """
    Visualizes real and reconstructed images side-by-side with their predictions and confidence scores.

    Args:
        real_images (np.ndarray): Array of real images.
        real_labels (np.ndarray): Array of true labels for real images.
        real_predictions (np.ndarray): Array of model predictions for real images.
        real_confidences (np.ndarray): Array of model confidences for real images.
        reconstructed_images (np.ndarray): Array of reconstructed images.
        reconstructed_labels (np.ndarray): Array of true labels for reconstructed images (target labels).
        reconstructed_predictions (np.ndarray): Array of model predictions for reconstructed images.
        reconstructed_confidences (np.ndarray): Array of model confidences for reconstructed images.
        target_names (list): A list of class names.
        target_class_indices (list): List of class indices for which reconstructions were generated.
    """
    n_classes = len(target_class_indices)
    fig, axes = plt.subplots(2, n_classes, figsize=(5 * n_classes, 10))

    for idx, class_idx in enumerate(target_class_indices):
        class_name = target_names[class_idx]

        # Top row: Real image (show first one)
        real_indices = np.where(real_labels == class_idx)[0]
        if len(real_indices) > 0:
            real_idx = real_indices[0]
            axes[0, idx].imshow(real_images[real_idx], cmap='gray')
            pred_name = target_names[real_predictions[real_idx]]
            is_correct = "✓" if real_predictions[real_idx] == real_labels[real_idx] else "✗"
            axes[0, idx].set_title(f'Real: {class_name}\nPred: {pred_name} {is_correct}\nConf: {real_confidences[real_idx]:.3f}',
                                   fontsize=10)
            axes[0, idx].axis('off')

        # Bottom row: Reconstructed image
        recon_idx = np.where(reconstructed_labels == class_idx)[0][0]
        axes[1, idx].imshow(reconstructed_images[recon_idx], cmap='gray')
        pred_name = target_names[reconstructed_predictions[recon_idx]]
        is_correct = "✓" if reconstructed_predictions[recon_idx] == reconstructed_labels[recon_idx] else "✗"
        axes[1, idx].set_title(f'Reconstructed: {class_name}\nPred: {pred_name} {is_correct}\nConf: {reconstructed_confidences[recon_idx]:.3f}',
                               fontsize=10, fontweight='bold')
        axes[1, idx].axis('off')

    plt.suptitle('Real vs Reconstructed Images Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

