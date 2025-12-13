"""
Evaluation metrics for model inversion attacks and reconstructions.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(a, b):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    SSIM measures the similarity between two images based on luminance,
    contrast, and structure. Higher values indicate more similar images.
    
    Args:
        a, b: Images in [0..1], shape HxW
    
    Returns:
        SSIM score (higher is better, max 1.0)
    """
    return ssim(a, b, data_range=1.0)


def compute_psnr(a, b):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    PSNR measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise. Higher values indicate better quality.
    
    Args:
        a, b: Images in [0..1], shape HxW
    
    Returns:
        PSNR in dB (higher is better)
    """
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def evaluate_attack_metrics(attack_results, y_test, X_test, nes_pop):
    """
    Evaluate attack metrics including confidence, queries, SSIM, and PSNR.
    
    This function computes comprehensive metrics for model inversion attacks,
    comparing reconstructed images against ground truth test images.
    
    Args:
        attack_results: Dictionary mapping class_idx to attack result dict containing:
            - 'class_name': Name of the target class
            - 'best_image': Best reconstructed image (numpy array)
            - 'final_confidence': Final confidence score for target class
            - 'history': List of best probabilities at each iteration
        y_test: Ground truth labels for test set (numpy array)
        X_test: Ground truth images from test set (numpy array)
        nes_pop: Population size used in NES attack (for calculating queries)
    
    Returns:
        attack_metrics: Dictionary mapping class_idx to metrics dict containing:
            - 'class_name': Name of the target class
            - 'final_confidence': Final confidence score
            - 'queries_used': Total number of queries used
            - 'iterations': Number of optimization iterations
            - 'ssim': SSIM score (or None if no ground truth available)
            - 'psnr': PSNR score (or None if no ground truth available)
    """
    attack_metrics = {}
    
    print("=" * 60)
    print("ATTACK METRICS EVALUATION")
    print("=" * 60)
    
    for class_idx, result in attack_results.items():
        class_name = result['class_name']
        best_image = result['best_image']
        final_confidence = result['final_confidence']
        history = result['history']
        queries_used = len(history) * nes_pop
        
        # Find a ground-truth image from the test set for comparison
        test_indices = np.where(y_test == class_idx)[0]
        if len(test_indices) > 0:
            # Use the first test image as reference
            ground_truth = X_test[test_indices[0]]
            
            # Compute SSIM and PSNR
            ssim_score = compute_ssim(best_image, ground_truth)
            psnr_score = compute_psnr(best_image, ground_truth)
        else:
            ssim_score = None
            psnr_score = None
        
        attack_metrics[class_idx] = {
            'class_name': class_name,
            'final_confidence': final_confidence,
            'queries_used': queries_used,
            'iterations': len(history),
            'ssim': ssim_score,
            'psnr': psnr_score
        }
        
        print(f"\n{class_name}:")
        print(f"  Final target class probability: {final_confidence:.4f}")
        print(f"  Queries used: {queries_used}")
        print(f"  Iterations: {len(history)}")
        if ssim_score is not None:
            print(f"  SSIM (vs ground truth): {ssim_score:.4f}")
            print(f"  PSNR (vs ground truth): {psnr_score:.2f} dB")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_confidence = np.mean([m['final_confidence'] for m in attack_metrics.values()])
    avg_queries = np.mean([m['queries_used'] for m in attack_metrics.values()])
    avg_ssim = np.mean([m['ssim'] for m in attack_metrics.values() if m['ssim'] is not None])
    avg_psnr = np.mean([m['psnr'] for m in attack_metrics.values() if m['psnr'] is not None])
    
    print(f"Average final confidence: {avg_confidence:.4f}")
    print(f"Average queries per attack: {avg_queries:.1f}")
    if not np.isnan(avg_ssim):
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
    print("=" * 60)
    
    return attack_metrics

