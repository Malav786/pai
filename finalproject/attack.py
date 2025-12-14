"""Model inversion attack using Natural Evolution Strategy (NES)."""

import numpy as np
import torch
from tqdm import trange


def softmax(x):
    """Compute softmax function for numerical stability."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def nes_optimize(decoder, query_fn, class_index, latent_dim=128, pop=50, sigma=0.1, 
                 lr=0.1, iters=500, device='cpu', latent_reg=0.0, sigma_anneal=True,
                 min_iters=100):
    """
    Optimize latent vector using NES to maximize target class probability.
    
    Args:
        decoder: PyTorch decoder (latent vector -> image)
        query_fn: Function to query the classifier
        class_index: Target class index
        latent_dim: Latent dimension
        pop: Population size per iteration
        sigma: Initial noise scale
        lr: Learning rate
        iters: Maximum optimization iterations
        device: 'cpu' or 'cuda'
        latent_reg: L2 regularization for latent vector
        sigma_anneal: Whether to reduce sigma over time
        min_iters: Minimum iterations before early stopping
    Returns:
        best_z: Best latent vector found
        history: Best probabilities per iteration
        best_image: Reconstructed image
    """
    # Initialize latent vector
    z = np.random.randn(latent_dim).astype(np.float32)
    best_z = z.copy()
    best_score = -np.inf
    best_image = None
    history = []

    initial_sigma = sigma

    for t in trange(iters, desc="NES Optimization"):
        # Progressive sigma annealing
        current_sigma = initial_sigma * (1.0 - 0.9 * (t / iters)) if sigma_anneal else sigma

        # Sample population of perturbations
        eps = np.random.randn(pop, latent_dim).astype(np.float32)
        scores = np.zeros(pop, dtype=np.float32)

        for i in range(pop):
            z_try = z + current_sigma * eps[i]

            # Decode latent vector to image
            with torch.no_grad():
                z_t = torch.from_numpy(z_try).unsqueeze(0).to(device)
                img_t = decoder(z_t).cpu().numpy()[0]
                img_for_query = np.clip(img_t.squeeze(), 0, 1)

            # Query classifier
            probs = query_fn(img_for_query)
            scores[i] = probs[class_index]

        # Standardize scores
        if scores.std() > 1e-8:
            A = (scores - scores.mean()) / (scores.std() + 1e-8)
        else:
            A = scores - scores.mean()

        # NES gradient estimate
        grad = (A.reshape(-1, 1) * eps).mean(axis=0) / current_sigma

        # Latent regularization
        if latent_reg > 0:
            grad = grad - latent_reg * z

        # Update latent vector
        z = z + lr * grad

        # Evaluate current latent
        with torch.no_grad():
            z_t = torch.from_numpy(z).unsqueeze(0).to(device)
            img_t = decoder(z_t).cpu().numpy()[0]
            img_for_eval = np.clip(img_t.squeeze(), 0, 1)

        cur_prob = query_fn(img_for_eval)[class_index]

        # Track best solution
        if cur_prob > best_score:
            best_score = cur_prob
            best_z = z.copy()
            best_image = img_for_eval.copy()

        history.append(cur_prob)

        # Early stopping with minimum iterations
        if t >= min_iters and best_score > 0.99:
            print(f"Early stopping triggered at iteration {t+1} with confidence {best_score:.4f}")
            break

    return best_z, history, best_image


def create_query_fn(model, device='cpu', num_classes=7):
    """Wrap classifier as a query function for black-box attacks."""
    model.eval()

    def query_fn(img_numpy):
        if len(img_numpy.shape) == 2:
            img_tensor = torch.from_numpy(img_numpy).unsqueeze(0).float()
        else:
            img_tensor = torch.from_numpy(img_numpy).float()

        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device)
        img_tensor = torch.clamp(img_tensor, 0, 1)

        with torch.no_grad():
            logits = model(img_tensor)
            logits_np = logits.cpu().numpy()[0]
            probs = softmax(logits_np)

        return probs

    return query_fn


def run_attack_suite(decoder, query_fn, target_names, target_classes, device,
                     nes_sigma, nes_pop, nes_iters, latent_dim,
                     lr, latent_reg, sigma_anneal, min_iters):
    """Run NES-based model inversion attack for multiple classes."""
    print(f"Attack Configuration:")
    print(f"  Population size: {nes_pop}")
    print(f"  Noise scale (sigma): {nes_sigma}")
    print(f"  Iterations: {nes_iters}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Total queries per attack: {nes_pop * nes_iters}")
    print("=" * 60)

    attack_results = {}

    for class_idx in target_classes:
        class_name = target_names[class_idx]
        print(f"\n{'='*60}")
        print(f"Attacking target class: {class_name} (index {class_idx})")
        print(f"{'='*60}")

        best_z, history, best_image = nes_optimize(
            decoder=decoder,
            query_fn=query_fn,
            class_index=class_idx,
            latent_dim=latent_dim,
            pop=nes_pop,
            sigma=nes_sigma,
            lr=lr,
            iters=nes_iters,
            device=device,
            latent_reg=latent_reg,
            sigma_anneal=sigma_anneal,
            min_iters=min_iters
        )

        final_probs = query_fn(best_image)
        final_confidence = final_probs[class_idx]

        attack_results[class_idx] = {
            'class_name': class_name,
            'best_z': best_z,
            'history': history,
            'best_image': best_image,
            'final_confidence': final_confidence,
            'final_probs': final_probs
        }

        print(f"\nFinal confidence for {class_name}: {final_confidence:.4f}")
        print(f"Final prediction: {target_names[np.argmax(final_probs)]}")
        print(f"Total queries used: {len(history) * nes_pop}")

    return attack_results
