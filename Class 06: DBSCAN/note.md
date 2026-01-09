# Class 06: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Motivation: Limitations of K-Means & GMM
*   **K-Means**: Assumes spherical clusters, sensitive to outliers, requires specifying K.
*   **GMM**: Assumes elliptical/Gaussian distributions, sensitive to initialization and outliers, requires specifying K.
*   **Problem**: Real-world data often has irregular shapes (e.g., moons, concentric circles) and noise.

## DBSCAN Overview
*   **Concept**: groups together points that are close to each other (dense regions) and marks points as outliers if they lie in low-density regions.
*   **Key Parameters**:
    1.  **Epsilon ($\epsilon$)**: The radius of the neighborhood around a data point.
    2.  **MinPts**: The minimum number of points required within the $\epsilon$-radius to consider a region "dense".

## Classification of Points
1.  **Core Point**: A point that has at least **MinPts** points within its $\epsilon$-neighborhood (including itself).
2.  **Border Point**: A point that has fewer than **MinPts** within its $\epsilon$-neighborhood but lies within the neighborhood of a **Core Point**.
3.  **Noise Point (Outlier)**: A point that is neither a Core Point nor a Border Point.

## Algorithm Steps
1.  Pick an unvisited point.
2.  Check if it's a **Core Point** (has $\ge$ MinPts within $\epsilon$).
    *   If **Yes**: Start a new cluster. Recursively add all directly reachable points (density-connected) to this cluster.
    *   If **No**: Mark it as **Noise** (temporarily; it might later be found as a Border Point of another cluster).
3.  Repeat until all points are visited.

## Advantages of DBSCAN
*   **No need to specify K**: The number of clusters is determined automatically.
*   **Arbitrary Shapes**: Can find clusters of any shape (e.g., snakes, moons).
*   **Robust to Outliers**: Explicitly handles noise/outliers.

## Disadvantages
*   **Varying Density**: struggles with clusters of varying densities (effectively needs different $\epsilon$).
*   **High Dimensionality**: "Curse of Dimensionality" makes distance measures less reliable.

---

## Code Implementation
**Notebook**: `colab.ipynb`
*   **Dataset**: `make_moons` (irregular shapes) with added noise.
*   **Comparison**:
    *   **K-Means**: Fails to separate the two moons correctly (draws linear boundaries).
    *   **GMM**: Also fails on the moon shapes.
    *   **DBSCAN**: Successfully identifies the two moon clusters and marks the random noise points as outliers (-1 label).
*   **Library**: `sklearn.cluster.DBSCAN`
