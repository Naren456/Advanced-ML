# Class 02: Dimensionality Reduction using tSNE

## PCA Recap & Limitations
*   **Assumptions of PCA**:
    *   Lineartiy: Assumes data lies on a lower-dimensional linear subspace.
    *   Large variances have important structure.
    *   Principal Components are orthogonal.
*   **Limitations**:
    1.  **Sensitive to Scale**: Requires data standardization (Mean=0, Std=1).
    2.  **Outliers**: PCA tries to capture high variance, so outliers can significantly skew the Principal Components.
    3.  **Linearity**: Fails to capture non-linear manifold structures (e.g., Swiss Roll dataset).
    4.  **Local Structure**: PCA preserves *global* structure (variance) but often fails to preserve *local* structure (neighborhoods).

---

## The Curse of Dimensionality
*   **Definition**: Refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) that do not occur in low-dimensional settings.
*   **Example**:
    *   **Distance Concentration**: As dimensions increase, the distance between the nearest and farthest points approaches zero. In high dimensions, "nearest neighbors" become less meaningful because all points are roughly equidistant.
    *   **Data Sparsity**: To maintain the same density of data as dimensions increase, the amount of data needed grows exponentially. A dataset that covers a 1D line well will be incredibly sparse in a 100D space.
    *   *Analogy*: Finding a needle in a haystack becomes exponentially harder as the haystack grows in dimensions, even if the needle size remains constant.

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Intuition
*   **Goal**: Preserve **local structure** (keep similar points close) while also revealing some global structure (clusters).
*   **Non-linear**: Able to capture complex non-linear manifolds.

### How it works (Internals)
1.  **High-Dimensional Space**:
    *   Computes similarity between points using a **Gaussian** distribution.
    *   Points closer together have a higher probability (similarity).
2.  **Low-Dimensional Map**:
    *   Computes similarity between points in the lower-dimensional space using a **Student's t-distribution** (heavier tails).
    *   Minimizes the divergence (Kullback-Leibler divergence) between the two probability distributions using Gradient Descent.

### Crowding Problem
*   In high dimensions, we have more room to place equidistant points. In lower dimensions, we run out of space ("crowding").
*   **Solution**: t-SNE uses the **Student's t-distribution** (with 1 degree of freedom) in the low-dimensional map. The heavier tails allow dissimilar points to be modeled as farther apart in the low-dimensional representation, effectively relieving the crowding.

---

## Code Implementation
**Notebook**: `Lec_2_PCA_+_tSNE.ipynb`

### Key Steps
1.  **Data Generation**: Creating synthetic weight/height data.
2.  **Scaling**: Demonstrates PCA differences with and without `StandardScaler`.
    *   *Unscaled*: Variance dominated by the feature with larger magnitude.
    *   *Scaled*: Variance shared/more balanced.
3.  **Digits Dataset**: Loading `sklearn.datasets.load_digits`.
4.  **Visualization**: Using `matplotlib` to display the digits.
