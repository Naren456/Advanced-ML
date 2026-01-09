# 🎤 QA & Viva: t-SNE & UMAP

### 🟢 Basic Questions
1.  **What is UMAP?**
    *   Uniform Manifold Approximation and Projection. It is a dimensionality reduction technique similar to t-SNE but faster and better at preserving global structure.

### 🟡 Intermediate Questions
2.  **Comparison: UMAP vs. t-SNE?**
    *   **Speed:** UMAP is significantly faster.
    *   **Structure:** UMAP preserves global structure (relationships between distant clusters) better than t-SNE.
    *   **Initialization:** UMAP is deterministic (if random state is set), whereas t-SNE is inherently stochastic.
3.  **Can we use t-SNE/UMAP for feature engineering in a classifier?**
    *   Yes, but with caution. UMAP is generally preferred because it provides a `transform` method for new data, whereas standard t-SNE does not handle new unseen data well.

### 🔴 Advanced Questions
4.  **How does UMAP handle topology?**
    *   It assumes the data lies on a manifold. It constructs a fuzzy simplicial set representation and optimizes the layout to preserve this topology in lower dimensions.
