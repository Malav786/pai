"""Model inversion attack using Natural Evolution Strategy (NES)."""

import numpy as np
import torch
from tqdm import trange


def softmax(x):
    """Compute softmax function for numerical stability."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def nes_optimize(decoder, query_fn, class_index, latent_dim, pop, sigma,
                lr, iters, device, latent_reg, sigma_anneal,
                min_iters, use_antithetic):
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
    decoder.eval()

    z = np.random.randn(latent_dim).astype(np.float32)
    best_z = z.copy()
    best_score = -np.inf
    best_image = None
    history = []

    initial_sigma = float(sigma)

    for t in trange(iters, desc="NES Optimization"):
        current_sigma = initial_sigma * (1.0 - 0.9 * (t / iters)) if sigma_anneal else initial_sigma

        # --- sample perturbations ---
        if use_antithetic:
            half = pop // 2
            eps_half = np.random.randn(half, latent_dim).astype(np.float32)
            eps = np.concatenate([eps_half, -eps_half], axis=0)
            # if pop is odd, add one extra random vector
            if eps.shape[0] < pop:
                eps = np.vstack([eps, np.random.randn(1, latent_dim).astype(np.float32)])
        else:
            eps = np.random.randn(pop, latent_dim).astype(np.float32)

        scores = np.zeros(eps.shape[0], dtype=np.float32)

        # --- evaluate population ---
        for i in range(eps.shape[0]):
            z_try = z + current_sigma * eps[i]

            with torch.no_grad():
                z_t = torch.from_numpy(z_try).unsqueeze(0).to(device)   # (1, latent_dim)
                img_t = decoder(z_t)                                    # (1,1,H,W)
                img_np = img_t.detach().cpu().numpy()
                img_for_query = np.clip(img_np[0, 0], 0.0, 1.0)         # (H,W)

            probs = query_fn(img_for_query)
            scores[i] = float(probs[class_index])

        # --- standardize scores (stable) ---
        std = scores.std()
        if std > 1e-8:
            A = (scores - scores.mean()) / (std + 1e-8)
        else:
            A = scores - scores.mean()

        # --- NES gradient estimate ---
        grad = (A.reshape(-1, 1) * eps).mean(axis=0) / (current_sigma + 1e-12)

        # --- latent L2 regularization (optional) ---
        if latent_reg > 0:
            grad = grad - latent_reg * z

        # --- update z ---
        z = z + lr * grad

        # --- evaluate current z ---
        with torch.no_grad():
            z_t = torch.from_numpy(z).unsqueeze(0).to(device)
            img_t = decoder(z_t)
            img_np = img_t.detach().cpu().numpy()
            img_for_eval = np.clip(img_np[0, 0], 0.0, 1.0)

        cur_prob = float(query_fn(img_for_eval)[class_index])

        if cur_prob > best_score:
            best_score = cur_prob
            best_z = z.copy()
            best_image = img_for_eval.copy()

        history.append(cur_prob)

        # Early stopping (after min_iters)
        if t >= min_iters and best_score > 0.95:
            print(f"Early stopping at iter {t+1}")
            break

    return best_z, history, best_image



def create_query_fn(model, device, num_classes):
    """Wrap classifier as a query function for black-box attacks."""
    model.eval()

    def to_model_tensor(img_numpy: np.ndarray) -> torch.Tensor:
        img = img_numpy

        # Accept: (H,W), (1,H,W), (H,W,1)
        if img.ndim == 2:
            img = img[None, None, ...]          # 1,1,H,W
        elif img.ndim == 3:
            if img.shape[0] == 1:               # 1,H,W
                img = img[None, ...]            # 1,1,H,W
            elif img.shape[-1] == 1:            # H,W,1
                img = img.transpose(2, 0, 1)[None, ...]  # 1,1,H,W
            else:
                raise ValueError("Expected grayscale image with 1 channel (H,W,1) or (1,H,W).")
        else:
            raise ValueError("Expected image as (H,W), (1,H,W), or (H,W,1).")

        t = torch.from_numpy(img).float()
        t = torch.clamp(t, 0.0, 1.0)
        return t

    def query_fn(img_numpy):
        img_tensor = to_model_tensor(img_numpy).to(device)

        with torch.no_grad():
            logits = model(img_tensor)  # (1, num_classes)
            if logits.shape[-1] != num_classes:
                raise ValueError(f"Expected {num_classes} classes, got {logits.shape[-1]}")
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return probs

    return query_fn


def run_attack_suite(decoder, query_fn, target_names, target_classes, device,
                     nes_sigma, nes_pop, nes_iters, latent_dim,
                     lr, latent_reg, sigma_anneal, min_iters):
    """Run NES-based model inversion attack for multiple classes."""
    print("Attack Configuration:")
    print(f"  Population size: {nes_pop}")
    print(f"  Noise scale (sigma): {nes_sigma}")
    print(f"  Iterations: {nes_iters}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Total queries per attack (upper bound): {nes_pop * nes_iters}")
    print("=" * 60)

    decoder = decoder.to(device).eval()

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
            min_iters=min_iters,
            use_antithetic=True
        )

        final_probs = query_fn(best_image)
        final_confidence = float(final_probs[class_idx])

        attack_results[class_idx] = {
            "class_name": class_name,
            "best_z": best_z,
            "history": history,
            "best_image": best_image,
            "final_confidence": final_confidence,
            "final_probs": final_probs,
            "total_queries_used": len(history) * nes_pop
        }

        print(f"\nFinal confidence for {class_name}")
        print(f"Prediction: {target_names[int(np.argmax(final_probs))]}")
        print(f"Total queries used: {len(history) * nes_pop}")

    return attack_results