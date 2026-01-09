# Class 01: Dimensionality Reduction using PCA

## Recap of Dimensionality Reduction
*   **Goal**: Reduce the number of features while retaining as much information (variance) as possible.
*   **Trade-off**: A small loss of information for a significant reduction in features.

## Principal Component Analysis (PCA)

### Properties of Principal Components (PCs)
1.  **Orthogonal**: The new axes (Principal Components) are perpendicular (orthogonal) to each other.
2.  **Number of PCs**: Initially equal to the number of original features, but we select the top $k$ components that explain the most variance.
3.  **Variance**: The spread of data along an axis. PCA aims to find axes with maximum variance.

### Mathematical Intuition
*   **Eigenvectors & Eigenvalues**:
    *   **Eigenvectors**: Determine the direction of the new feature space (the axis).
    *   **Eigenvalues**: Determine the magnitude (variance) of the data along that eigenvector.
*   **Covariance Matrix**: Used to calculate eigenvectors and eigenvalues.
*   **Steps**:
    1.  Standardize the data (Mean centering and scaling).
    2.  Compute the Covariance Matrix.
    3.  Compute Eigenvectors and Eigenvalues of the Covariance Matrix.
    4.  Sort Eigenvalues in descending order and select the top $k$ Eigenvectors.
    5.  Transform the original data onto the new subspace.

### Information Loss
*   Dimensionality reduction involves some loss of information.
*   We aim to minimize this loss by choosing PCs with the highest eigenvalues (variance).

---

## Code Implementation: MNIST Dataset
**Notebook**: `Lec_1_PCA.ipynb`

### 1. Dataset Overview
*   **MNIST**: 28x28 grayscale images of handwritten digits (0-9).
*   **Data Structure**: Each image is a 784-dimensional vector (28*28 pixels).
*   **Shape**: (42000, 784) for train data (excluding label).

### 2. Preprocessing
*   **Loading Data**: `pd.read_csv('mnist_train.csv')`
*   **Standardization**:
    *   Essential step before PCA.
    *   `StandardScaler` from `sklearn.preprocessing`.
    *   `X_stand = scaler.transform(X)`

### 3. Implementing PCA
*   **Library**: `sklearn.decomposition.PCA`
*   **Initialization**: `pca = PCA(n_components = 2)` (Reducing to 2 dimensions for visualization)
*   **Transformation**: `X_embedded = pca.fit_transform(X_stand)`
*   **Result**: Shape changes from (1797, 64) -> (1797, 2) *[Note: Notebook uses `digits` dataset later which has 64 features]*

### 4. Visualization
*   The notebook demonstrates reshaping 1D vectors back to 2D images (`reshape(28, 28)`) for display.
*   Grayscale color map (`cmap="gray"`) is used.
