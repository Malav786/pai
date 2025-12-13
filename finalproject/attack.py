"""Model inversion attack using Natural Evolution Strategy (NES)."""

import numpy as np
import torch
from tqdm import trange


def softmax(x):
    """Compute softmax function for numerical stability.
    
    Args:
        x: Input array (can be 1D or 2D)
    
    Returns:
        Softmax probabilities with same shape as input
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()


def nes_optimize(decoder, query_fn, class_index, latent_dim=128, pop=50, sigma=0.1, 
                 lr=0.1, iters=500, device='cpu', latent_reg=0.0, sigma_anneal=True):
    """Optimize latent vector using Natural Evolution Strategy (NES) to maximize target class probability.
    
    This function performs a black-box model inversion attack by optimizing a latent vector
    that, when decoded, produces an image that maximizes the target classifier's confidence
    for a specific class. The attack uses NES to estimate gradients without accessing
    the classifier's internal parameters.
    
    Args:
        decoder: PyTorch module that maps z (tensor shape (B, latent_dim)) -> images tensor (B,C,H,W)
        query_fn: Function that takes image_numpy (H x W or C x H x W) and returns softmax probabilities
        class_index: Target class index to maximize
        latent_dim: Dimension of latent space. Default is 128.
        pop: Population size (number of samples per iteration). Default is 50.
        sigma: Noise scale for perturbations. Default is 0.1.
        lr: Learning rate for gradient updates. Default is 0.1.
        iters: Number of optimization iterations. Default is 500.
        device: Device to run on ('cpu' or 'cuda'). Default is 'cpu'.
        latent_reg: L2 regularization coefficient for latent vector (penalize distance from prior).
                    Default is 0.0 (no regularization).
        sigma_anneal: Whether to use progressive sigma annealing. Default is True.
    
    Returns:
        best_z: Best latent vector found (numpy array of shape (latent_dim,))
        history: List of best probabilities at each iteration
        best_image: Best reconstructed image (numpy array)
    """
    # Initialize latent vector from standard normal distribution
    z = np.random.randn(latent_dim).astype(np.float32)
    best_z = z.copy()
    best_score = -np.inf
    best_image = None
    history = []
    
    # Initial sigma for annealing
    initial_sigma = sigma
    
    for t in trange(iters, desc="NES Optimization"):
        # Progressive sigma annealing: start large, shrink over time
        if sigma_anneal:
            # Linear annealing: sigma decreases from initial_sigma to 0.1 * initial_sigma
            current_sigma = initial_sigma * (1.0 - 0.9 * (t / iters))
        else:
            current_sigma = sigma
        
        # Sample population of perturbations
        eps = np.random.randn(pop, latent_dim).astype(np.float32)
        scores = np.zeros(pop, dtype=np.float32)
        
        # Evaluate each perturbation
        for i in range(pop):
            z_try = z + current_sigma * eps[i]
            
            # Decode to image via decoder (torch)
            with torch.no_grad():
                z_t = torch.from_numpy(z_try).unsqueeze(0).to(device)
                img_t = decoder(z_t).cpu().numpy()[0]  # shape (C,H,W)
                # Convert to HxW grayscale image array [0..1]
                img_for_query = img_t.squeeze()  # Remove channel dimension if present
                # Ensure values are in [0, 1] range
                img_for_query = np.clip(img_for_query, 0, 1)
            
            # Query the classifier
            probs = query_fn(img_for_query)
            scores[i] = probs[class_index]
        
        # Standardize scores (fitness shaping)
        if scores.std() > 1e-8:
            A = (scores - scores.mean()) / (scores.std() + 1e-8)
        else:
            A = scores - scores.mean()
        
        # NES gradient estimate
        grad = (A.reshape(-1, 1) * eps).mean(axis=0) / current_sigma
        
        # Apply latent regularization (penalize large deviations from prior)
        if latent_reg > 0:
            grad = grad - latent_reg * z
        
        # Update latent vector
        z = z + lr * grad
        
        # Evaluate current z
        with torch.no_grad():
            z_t = torch.from_numpy(z).unsqueeze(0).to(device)
            img_t = decoder(z_t).cpu().numpy()[0]
            img_for_eval = img_t.squeeze()
            img_for_eval = np.clip(img_for_eval, 0, 1)
        
        cur_prob = query_fn(img_for_eval)[class_index]
        
        # Track best solution
        if cur_prob > best_score:
            best_score = cur_prob
            best_z = z.copy()
            best_image = img_for_eval.copy()
        
        history.append(cur_prob)
        
        # Early stopping if we've achieved high confidence
        if best_score > 0.99:
            break
    
    return best_z, history, best_image


def create_query_fn(model, device='cpu', num_classes=7):
    """Create a query function that wraps the target classifier.
    
    Args:
        model: PyTorch model (classifier) to query
        device: Device to run on ('cpu' or 'cuda'). Default is 'cpu'.
        num_classes: Number of classes. Default is 7.
    
    Returns:
        query_fn: Function that takes numpy image (H x W) and returns softmax probabilities
    """
    model.eval()
    
    def query_fn(img_numpy):
        """
        Query function for black-box attack.
        
        Args:
            img_numpy: H x W array in [0..1] range
        
        Returns:
            1D numpy array of softmax probabilities (len=num_classes)
        """
        # Convert to torch tensor
        if len(img_numpy.shape) == 2:
            # Add channel dimension: (H, W) -> (1, H, W)
            img_tensor = torch.from_numpy(img_numpy).unsqueeze(0).float()
        else:
            img_tensor = torch.from_numpy(img_numpy).float()
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Move to device
        img_tensor = img_tensor.to(device)
        
        # Ensure values are in [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(img_tensor)
            # Convert to numpy and apply softmax
            logits_np = logits.cpu().numpy()[0]
            probs = softmax(logits_np)
        
        return probs
    
    return query_fn


def run_attack_suite(decoder, query_fn, target_names, target_classes, device='cpu',
                     nes_sigma=0.1, nes_pop=50, nes_iters=800, latent_dim=128,
                     lr=0.1, latent_reg=0.0, sigma_anneal=True):
    """Run model inversion attacks for multiple target classes.
    
    This function executes NES-based model inversion attacks for a list of target classes,
    collecting results for each attack including reconstructed images, confidence scores,
    and optimization history.
    
    Args:
        decoder: PyTorch decoder module that maps latent vectors to images
        query_fn: Function that takes an image and returns softmax probabilities
        target_names: List of class names (for display purposes)
        target_classes: List of class indices to attack (e.g., [0, 1, 3])
        device: Device to run on ('cpu' or 'cuda'). Default is 'cpu'.
        nes_sigma: Noise scale for perturbations. Default is 0.1.
        nes_pop: Population size (samples per iteration). Default is 50.
        nes_iters: Number of optimization iterations. Default is 800.
        latent_dim: Latent dimension of autoencoder. Default is 128.
        lr: Learning rate for gradient updates. Default is 0.1.
        latent_reg: L2 regularization coefficient for latent vector. Default is 0.0.
        sigma_anneal: Whether to use progressive sigma annealing. Default is True.
    
    Returns:
        attack_results: Dictionary mapping class_idx to attack result dict containing:
            - 'class_name': Name of the target class
            - 'best_z': Best latent vector found
            - 'history': List of best probabilities at each iteration
            - 'best_image': Best reconstructed image (numpy array)
            - 'final_confidence': Final confidence score for target class
            - 'final_probs': Final softmax probability vector
    """
    import numpy as np
    
    # Print attack configuration
    print(f"Attack Configuration:")
    print(f"  Population size: {nes_pop}")
    print(f"  Noise scale (sigma): {nes_sigma}")
    print(f"  Iterations: {nes_iters}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Total queries per attack: {nes_pop * nes_iters}")
    print("=" * 60)
    
    attack_results = {}
    
    # Run attack for each target class
    for class_idx in target_classes:
        class_name = target_names[class_idx]
        print(f"\n{'='*60}")
        print(f"Attacking target class: {class_name} (index {class_idx})")
        print(f"{'='*60}")
        
        # Run NES optimization
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
            sigma_anneal=sigma_anneal
        )
        
        # Evaluate final result
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

