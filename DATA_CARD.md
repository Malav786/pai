# Data Card

## Dataset: Labeled Faces in the Wild (LFW)

### 1. Where the data came from

The dataset is accessed through scikit-learn's data loading utilities:

- **Source**: [scikit-learn datasets documentation](https://scikit-learn.org/0.19/datasets/labeled_faces.html)
- **Original dataset**: [Labeled Faces in the Wild official website](http://vis-www.cs.umass.edu/lfw/)
- **Loading method**: `sklearn.datasets.fetch_lfw_people()`

### 2. Purpose of the dataset

The Labeled Faces in the Wild (LFW) dataset is designed to support research in face recognition under real-world, unconstrained conditions such as varying lighting, pose, and expressions.

It serves as a standard benchmark for evaluating and comparing the robustness and generalization of facial recognition models.

LFW is also widely used in privacy and security research to study inference attacks, data leakage, and adversarial behavior in biometric systems.

### 3. Dataset authors

The Labeled Faces in the Wild (LFW) dataset was authored by:
- Gary B. Huang
- Manu Ramesh
- Tamara Berg
- Erik Learned-Miller

**Institution**: University of Massachusetts Amherst

**Reference**: 
> Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007). Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.

### 4. Dataset ownership

The Labeled Faces in the Wild (LFW) dataset is owned and maintained by the University of Massachusetts Amherst, specifically the research group led by Erik Learned-Miller.

### 5. License

The Labeled Faces in the Wild (LFW) dataset is distributed for **non-commercial research and educational use only**.

It does not have a standard open-source license (such as MIT or Apache) and is subject to usage terms defined by the University of Massachusetts Amherst.

### 6. Data processing

**To simplify the dataset for this project**, we filtered the data to include only people with at least 70 faces per person (`min_faces_per_person=70`). This reduces the dataset from thousands of people to 7 people, but ensures sufficient samples per person for training and evaluation.

Beyond this filtering, no significant processing was performed on the data beyond the standard preprocessing provided by scikit-learn's `fetch_lfw_people()` function, which includes:
- Automatic download and caching
- JPEG decoding
- Resizing (as specified by the `resize` parameter, set to 0.4)
- Conversion to numpy arrays
- Filtering by minimum faces per person (set to 70 for this project)

### Dataset details (as used in this project)

- **Minimum faces per person**: 70
- **Resize factor**: 0.4
- **Number of samples**: 1,288 images
- **Image dimensions**: 50 × 37 pixels (grayscale)
- **Number of classes**: 7 people
- **Feature dimensions**: 1,850 (flattened: 50 × 37)
