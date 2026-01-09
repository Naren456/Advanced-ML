# Class 05: Gaussian Mixture Models (GMM)

## Clustering Types
*   **Hard Clustering** (e.g., K-Means): Each data point determines to strictly one cluster.
*   **Soft Clustering** (e.g., GMM): Each data point has a **probability** of belonging to each cluster.

## Gaussian Distribution Recap
*   **1D Gaussian**: Defined by Mean ($\mu$) and Standard Deviation ($\sigma$).
    *   **Mean ($\mu$)**: Position/Center of the distribution.
    *   **Standard Deviation ($\sigma$)**: Spread/Width of the distribution.
*   **2D Gaussian**: Defined by Mean vector ($\mu$), Covariance Matrix ($\Sigma$).
    *   Describes the shape (spherical, elliptical) and orientation.

## Gaussian Mixture Models (GMM)
*   **Concept**: Assumes the data is generated from a mixture of several Gaussian distributions.
*   **Goal**: Find the parameters ($\mu, \Sigma$, and mixing coefficients) that best explain the data.
*   **Algorithm**: **Expectation-Maximization (EM)**.

### Expectation-Maximization (EM) Algorithm
1.  **Initialization**: Randomly initialize the parameters (Mean, Variance, and mixing weights) for **K** Gaussians.
2.  **Expectation Step (E-Step)**:
    *   Calculate the **likelihood** (probability) of each data point belonging to each of the K Gaussian clusters.
    *   This results in a matrix of probabilities (Responsibility).
3.  **Maximization Step (M-Step)**:
    *   Update the parameters ($\mu, \Sigma$, weights) for each Gaussian.
    *   New Mean = Weighted average of data points (weights = probabilities from E-step).
    *   New Variance = Weighted variance.
4.  **Convergence**: Repeat E-Step and M-Step until the log-likelihood of the data converges (stops increasing).

## GMM vs. K-Means

| Feature | K-Means | GMM |
| :--- | :--- | :--- |
| **Clustering Type** | Hard Clustering (0 or 1) | Soft Clustering (Probabilities) |
| **Cluster Shape** | Spherical (Circular) | Elliptical (Flexible shapes) |
| **Cluster Size** | Tends to find equal-sized clusters | Can handle varying cluster sizes |
| **Parameters** | Centroids | Mean & Variance (Covariance) |
| **Distance Metric** | Euclidean Distance | Mahalanobis Distance (Probabilistic) |
