from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.utils.data as data

def load_lfw(min_faces_per_person=70, resize=0.4):
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                            resize=resize)
    X = lfw.images # shape (n_samples, h, w)
    y = lfw.target
    target_names = lfw.target_names
    return X.astype('float32')/255.0, y, target_names

def create_splits(X, y, test_size, val_size, random_state):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, stratify=y, random_state=random_state)
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

class NumpyFaceDataset(data.Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = (self.X[idx]*255).astype('uint8')
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.y is None:
            return img
        return img, int(self.y[idx]) 