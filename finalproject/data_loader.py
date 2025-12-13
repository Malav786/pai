"""Data loader module for LFW (Labeled Faces in the Wild) dataset."""

import os
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import fetch_lfw_people


class LFWDataLoader:
    """Loader for the Labeled Faces in the Wild (LFW) dataset.
    
    This class provides a convenient interface to load and access the LFW dataset
    from scikit-learn, with configurable parameters for minimum faces per person
    and image resizing.
    
    Attributes:
        data_home: Path where the dataset is stored/cached
        min_faces_per_person: Minimum number of faces per person (default: 70)
        resize: Resize factor for images (default: 0.4)
        dataset: The loaded dataset object from sklearn
    """
    
    def __init__(
        self,
        data_home: str = '../dataset/',
        min_faces_per_person: int = 70,
        resize: float = 0.4,
        verbose: bool = True
    ):
        """Initialize the LFW data loader.
        
        Args:
            data_home: Directory path where dataset will be saved/cached
            min_faces_per_person: Minimum number of faces per person to include
            resize: Resize factor for images (0.4 means 40% of original size)
            verbose: Whether to print information about dataset location
        """
        self.data_home = data_home
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize
        self.verbose = verbose
        self.dataset = None
        
        if self.verbose:
            self._print_dataset_info()
    
    def _print_dataset_info(self) -> None:
        """Print information about where the dataset will be saved."""
        lfw_path = os.path.join(self.data_home, 'lfw_home')
        abs_path = os.path.abspath(lfw_path)
        
        print(f"Dataset will be saved to: {self.data_home}")
        if os.path.exists(lfw_path):
            print(f"✓ Dataset directory already exists")
            print(f"  Location: {abs_path}")
        else:
            print(f"⚠ Dataset will be downloaded to: {abs_path}")
            print(f"  (Directory will be created if it doesn't exist)")
    
    def load(self) -> None:
        """Load the LFW dataset.
        
        Downloads the dataset if not already cached, then loads it into memory.
        """
        self.dataset = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=self.resize,
            data_home=self.data_home
        )
    
    @property
    def images(self) -> np.ndarray:
        """Get the 2D image array.
        
        Returns:
            3D numpy array of shape (n_samples, height, width)
            Use this for visualization with matplotlib
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.dataset.images
    
    @property
    def data(self) -> np.ndarray:
        """Get the flattened feature vectors.
        
        Returns:
            2D numpy array of shape (n_samples, n_features)
            Use this as input (X) for machine learning models
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.dataset.data
    
    @property
    def target(self) -> np.ndarray:
        """Get the target labels (person IDs).
        
        Returns:
            1D numpy array of shape (n_samples,)
            Use this as target (y) for classification
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.dataset.target
    
    @property
    def target_names(self) -> np.ndarray:
        """Get the names of the people in the dataset.
        
        Returns:
            1D numpy array of person names
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.dataset.target_names
    
    def get_shape_info(self) -> dict:
        """Get shape information for all data arrays.
        
        Returns:
            Dictionary with shapes of images, data, and target
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        return {
            'images_shape': self.images.shape,
            'data_shape': self.data.shape,
            'target_shape': self.target.shape,
            'n_classes': len(self.target_names)
        }
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of images per person.
        
        Returns:
            Dictionary mapping person names to number of images
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        distribution = {}
        for i, name in enumerate(self.target_names):
            count = np.sum(self.target == i)
            distribution[name] = int(count)
        
        return distribution
    
    def get_person_indices(self, person_name: str) -> np.ndarray:
        """Get indices of all images for a specific person.
        
        Args:
            person_name: Name of the person (must match target_names)
            
        Returns:
            Array of indices where this person's images are located
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        if person_name not in self.target_names:
            raise ValueError(
                f"Person '{person_name}' not found. "
                f"Available names: {list(self.target_names)}"
            )
        
        person_id = np.where(self.target_names == person_name)[0][0]
        indices = np.where(self.target == person_id)[0]
        return indices

    def show_distinct_people(
        self,
        n_cols: int = 4,
        figsize: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = 42
    ) -> None:
        """Show 1 random image per distinct person (identity).

        Args:
            n_cols: Number of columns for the grid
            figsize: Figure size. If None, computed automatically.
            seed: Random seed for reproducibility
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        rng = np.random.default_rng(seed)
        n_people = len(self.target_names)

        # pick one random image index for each person id
        chosen = []
        for person_id in range(n_people):
            person_indices = np.where(self.target == person_id)[0]
            chosen.append(int(rng.choice(person_indices)))
        chosen = np.array(chosen)

        n_rows = int(np.ceil(n_people / n_cols))
        if figsize is None:
            figsize = (3 * n_cols, 3.5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for ax, idx in zip(axes, chosen):
            ax.imshow(self.images[idx], cmap="gray")
            ax.set_title(self.target_names[self.target[idx]], fontsize=9)
            ax.axis("off")

        for ax in axes[len(chosen):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

def load_lfw_data(
    data_home: str = '../dataset/',
    min_faces_per_person: int = 70,
    resize: float = 0.4,
    verbose: bool = True
) -> LFWDataLoader:
    """Convenience function to load LFW dataset.
    
    This is a simple wrapper that creates a LFWDataLoader, loads the data,
    and returns the loader instance ready to use.
    
    Args:
        data_home: Directory path where dataset will be saved/cached
        min_faces_per_person: Minimum number of faces per person to include
        resize: Resize factor for images (0.4 means 40% of original size)
        verbose: Whether to print information about dataset location
        
    Returns:
        LFWDataLoader instance with dataset already loaded
        
    Example:
        >>> loader = load_lfw_data()
        >>> images = loader.images
        >>> data = loader.data
        >>> target = loader.target
    """
    loader = LFWDataLoader(
        data_home=data_home,
        min_faces_per_person=min_faces_per_person,
        resize=resize,
        verbose=verbose
    )
    loader.load()
    return loader


