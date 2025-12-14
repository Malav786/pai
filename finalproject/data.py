from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.utils.data as data

from finalproject.data_loader import LFWDataLoader


def load_lfw(loader=None, min_faces_per_person=70, resize=0.4, data_home='../dataset/'):
    """Load and preprocess the LFW (Labeled Faces in the Wild) dataset.
    
    Uses an existing LFWDataLoader instance or creates a new one, then returns
    normalized numpy arrays ready for training. This is a convenience wrapper
    that normalizes pixel values to [0, 1] range.
    
    Args:
        loader (LFWDataLoader, optional): Existing loader instance to use.
            If None, a new loader will be created with the provided parameters.
        min_faces_per_person: Minimum number of faces per person to include.
            Default is 70. Only used if loader is None.
        resize: Resize factor for images. Default is 0.4 (40% of original size).
            Only used if loader is None.
        data_home: Directory path where dataset will be saved/cached.
            Default is '../dataset/'. Only used if loader is None.
    
    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Image data with shape (n_samples, height, width).
              Pixel values are normalized to [0, 1] range (float32).
            - y (np.ndarray): Target labels with shape (n_samples,).
              Each value is an integer index corresponding to a person.
            - target_names (np.ndarray): Array of person names corresponding
              to the label indices.
    
    Example:
        >>> # Using existing loader
        >>> loader = load_lfw_data()
        >>> X, y, target_names = load_lfw(loader=loader)
        
        >>> # Creating new loader
        >>> X, y, target_names = load_lfw(min_faces_per_person=70, resize=0.4)
    """
    if loader is None:
        loader = LFWDataLoader(
            data_home=data_home,
            min_faces_per_person=min_faces_per_person,
            resize=resize,
            verbose=False
        )
        loader.load()
    elif loader.dataset is None:
        # Loader exists but data not loaded yet
        loader.load()
    
    # sklearn's fetch_lfw_people already returns images in [0, 1] range (float)
    # Just convert to float32, no need to divide by 255
    X = loader.images.astype('float32')
    # Ensure values are in [0, 1] range (they should already be)
    X = np.clip(X, 0.0, 1.0)
    y = loader.target
    target_names = loader.target_names
    return X, y, target_names

def create_splits(X, y, test_size, val_size, random_state):
    """Split dataset into training, validation, and test sets.
    
    Creates three stratified splits of the data to ensure balanced class
    distribution across all sets. First splits into train and temp (val+test),
    then splits temp into validation and test sets.
    
    Args:
        X (np.ndarray): Feature data with shape (n_samples, ...).
        y (np.ndarray): Target labels with shape (n_samples,).
        test_size (float): Proportion of dataset to use for test set.
            Should be between 0.0 and 1.0.
        val_size (float): Proportion of dataset to use for validation set.
            Should be between 0.0 and 1.0.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training labels.
            - X_val (np.ndarray): Validation features.
            - y_val (np.ndarray): Validation labels.
            - X_test (np.ndarray): Test features.
            - y_test (np.ndarray): Test labels.
    
    Example:
        >>> X_train, y_train, X_val, y_val, X_test, y_test = create_splits(
        ...     X, y, test_size=0.2, val_size=0.1, random_state=42
        ... )
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, stratify=y, random_state=random_state)
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

class NumpyFaceDataset(data.Dataset):
    """PyTorch Dataset class for LFW face images stored as numpy arrays.
    
    This dataset class wraps numpy arrays of face images and makes them
    compatible with PyTorch DataLoader. Images are converted from normalized
    float arrays to PIL Images, and optionally transformed.
    
    Attributes:
        X (np.ndarray): Image data with shape (n_samples, height, width).
            Pixel values should be normalized to [0, 1] range.
        y (np.ndarray, optional): Target labels with shape (n_samples,).
            If None, dataset returns only images (useful for inference).
        transform (callable, optional): Optional transform to be applied
            on a sample. Should be a torchvision transform or callable.
    
    Example:
        >>> dataset = NumpyFaceDataset(X_train, y_train, transform=transforms.ToTensor())
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for images, labels in dataloader:
        ...     # images shape: (32, 1, 50, 37)
        ...     # labels shape: (32,)
        ...     pass
    """
    
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get the image array (should be in [0, 1] range)
        img_array = self.X[idx].copy()
        
        # If transform is provided, we need to convert to PIL Image first
        if self.transform:
            # Ensure values are in valid range [0, 1]
            img_array = np.clip(img_array, 0.0, 1.0)
            # Convert to [0, 255] uint8 for PIL Image
            img_uint8 = (img_array * 255.0).astype('uint8')
            # Create PIL Image in grayscale mode ('L')
            img = Image.fromarray(img_uint8, mode='L')
            # Apply transform (ToTensor() will convert back to [0, 1] and add channel dim)
            img = self.transform(img)
        else:
            # No transform: convert directly to tensor
            # Add channel dimension and convert to tensor
            img = torch.from_numpy(img_array).float().unsqueeze(0)  # Shape: (1, H, W)
        
        if self.y is None:
            return img
        return img, int(self.y[idx]) 