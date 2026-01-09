# Class 08: Anomaly Detection (Part 2)

## 1. Isolation Forest
*   **Concept**: An ensemble based algorithm that explicitly isolates anomalies instead of profiling normal points.
*   **Intuition**: Anomalies are "few and different". They are more susceptible to isolation than normal points.
    *   In a random tree, anomalies tend to have **shorter path lengths** (closer to the root) because they are easier to separate.
    *   Normal points are deeper in the tree (longer path lengths).
*   **Algorithm Steps**:
    1.  Select a feature randomly.
    2.  Select a split value randomly between the max and min values of the selected feature.
    3.  Partition the data recursively.
    4.  Repeat to build an ensemble of trees (Isolation Forest).
*   **Anomaly Score**:
    $$s(x, n) = 2^{- \frac{E(h(x))}{c(n)}}$$
    *   $E(h(x))$: Average path length of data point $x$ over all trees.
    *   $c(n)$: Average path length of unsuccessful search in BST given $n$ nodes (normalization factor).
*   **Interpretation**:
    *   $s \to 1$: **Anomaly** (Path length is much smaller than average).
    *   $s \to 0$: **Normal** (Path length is much larger than average).
    *   $s \approx 0.5$: Normal (Path length is similar to average).
*   **Parameters**: `contamination` (proportion of outliers in the dataset).

## 2. Local Outlier Factor (LOF)
*   **Concept**: A density-based method that compares the local density of a point with the local densities of its neighbors.
*   **Key Idea**: An outlier typically has a significantly lower local density than its $k$-nearest neighbors.
*   **LOF Score**:
    *   **LOF $\approx$ 1**: **Normal**. The point has a similar density to its neighbors.
    *   **LOF > 1**: **Anomaly**. The point has a lower density than its neighbors (it is isolated).
    *   **LOF < 1**: **Normal/Dense Region**. The point has a higher density than its neighbors.
*   **Parameters**:
    *   `n_neighbors`: Number of neighbors to consider (roughly corresponds to min points in DBSCAN context).
    *   `contamination`: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.

## Code Implementation
**Notebook**: `colab.ipynb`
*   **Library**: `sklearn.ensemble.IsolationForest`
*   **Dataset**: `AnomalyDetection.csv` (Mileage vs Price).
*   **Implementation**:
    *   Model Initialization: `model = IsolationForest(contamination=0.06, random_state=42)`
    *   Fit & Predict: `y = model.fit_predict(df)`
    *   Output: `1` for inliers, `-1` for outliers.
    *   Visualization: Scatter plot showing clusters with identified outliers marked.
