# Class 04: Clustering - KMeans & KMeans++

## Clustering Overview
*   **Unsupervised Learning**: Data has no labels. The goal is to group similar data points together.
*   **Goal**: Maximize intra-cluster similarity (minimize distance within clusters) and minimize inter-cluster similarity (maximize distance between clusters).

---

## K-Means Clustering

### Algorithm Steps
1.  **Initialize**: Choose **K** random centroids.
2.  **Calculate Distance**: Compute the distance of each point from all centroids.
3.  **Assign**: Assign each data point to its nearest centroid.
4.  **Re-calculate**: Compute the new centroid of each cluster (mean of points in the cluster).
5.  **Repeat**: Repeat steps 2-4 until convergence (centroids do not change).

### Evaluation Metrics (Choosing K)
1.  **Inertia / WCSS (Within-Cluster Sum of Squares)**:
    *   Sum of squared distances between each point and its assigned centroid.
    *   Lower is better, but it decreases as K increases.
2.  **Elbow Curve**:
    *   Plot Inertia vs. K.
    *   Look for the "elbow" point where the rate of decrease slows down significantly. This is typically the optimal K.
3.  **Silhouette Score**:
    *   Measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).
    *   Range: -1 to +1.
    *   Values closer to +1 indicate better defined clusters.

### Limitations of K-Means
1.  **Outliers**: Not robust to outliers; they can significantly shift centroids.
2.  **Shape**: Assumes spherical clusters; fails with complex shapes (e.g., concentric circles).
3.  **Initialization**: Random initialization can lead to poor convergence or getting stuck in local optima.

---

## K-Means++
*   **Problem Solved**: Addresses the random initialization issue of standard K-Means.
*   **Idea**: Instead of picking all centroids randomly, pick the first one randomly, then pick subsequent centroids such that they are **far away** from already chosen centroids.
*   **Benefit**: Better chances of finding the global optimum and often faster convergence.

---

## Code Implementation
**Notebook**: `colab.ipynb`
*   **Data**: E-commerce dataset (Customer segmentation).
*   **Preprocessing**: Scaling data using `StandardScaler`.
*   **Implementation**:
    *   Manual implementation of K-Means steps (initialize, assign, update).
    *   `sklearn.cluster.KMeans` usage.
    *   Visualizing clusters using `matplotlib`.
    *   Finding K using Elbow Method (plotting WCSS) and Silhouette Coefficients.
