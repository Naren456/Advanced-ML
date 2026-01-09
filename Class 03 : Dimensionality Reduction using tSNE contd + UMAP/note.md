# Class 03: Dimensionality Reduction using t-SNE (Contd.) & UMAP

## t-SNE Deep Dive

### Mathematical Internals
1.  **High-Dimensional Similarity ($P_{j|i}$)**:
    *   Calculated using a **Gaussian Distribution** centered at $x_i$.
    *   $P_{j|i}$ represents the conditional probability that $x_i$ would pick $x_j$ as its neighbor.
    *   Proportional to the probability density under the Gaussian.
2.  **Low-Dimensional Similarity ($Q_{j|i}$)**:
    *   Calculated using a **Student's t-distribution** (with 1 degree of freedom).
    *   Heavy tails of t-distribution help resolve the **Crowding Problem** (allowing dissimilar points to be further apart).
3.  **Cost Function (KL Divergence)**:
    *   We want the low-dimensional distribution $Q$ to match the high-dimensional distribution $P$.
    *   **Kullback-Leibler (KL) Divergence** measures the difference between $P$ and $Q$.
    *   Algorithm minimizes this divergence using Gradient Descent.

### Hyperparameters
*   **Perplexity**:
    *   Roughly implies the **number of meaningful neighbors** for each point.
    *   Controls the balance between local and global aspects of the data.
    *   Typical range: 5 to 50.
    *   Low perplexity -> Focus on very local structure.
    *   High perplexity -> Takes more global structure into account.
*   **Iterations (n_iter)**: Number of steps for gradient descent optimization. Typically 1000-5000.

### Limitations of t-SNE
1.  **Slow**: $O(N^2)$ complexity makes it slow for very large datasets.
2.  **Stochastic**: Results can vary between runs (non-deterministic).
3.  **Global Structure**: Preserves local structure well but often distorts global geometry (cluster distances might not differ meaningfully).
4.  **Hyperparameter Sensitivity**: Output heavily depends on Perplexity and Learning Rate.

---

## UMAP (Uniform Manifold Approximation and Projection)
*   **Overview**: A newer dimensionality reduction technique (2018) often preferred over t-SNE.
*   **Key Advantages over t-SNE**:
    1.  **Faster**: Much more scalable to large datasets.
    2.  **Global Structure**: Better at preserving global structure (distances between clusters) while still maintaining local structure.
    3.  **Deterministic**: Can be made deterministic.
*   **How it works**: Uses manifold theory and topological data analysis to construct a fuzzy topological representation of the high-dimensional data and then finds a low-dimensional topological representation that matches it.

---

## Code Implementation
**Notebook**: `colab.ipynb`
*   The notebook currently shows basic PCA implementation steps (Standardization, PCA calculation) similar to previous classes, serving as a refresher or base for comparison.
