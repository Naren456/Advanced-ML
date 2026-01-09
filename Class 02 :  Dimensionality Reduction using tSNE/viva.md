# 🎤 Real-World Interview Viva: t-SNE

### 🏢 Google / Airbnb
1.  **Q:** Why is t-SNE better than PCA for visualization?
    *   **A:** PCA is linear and focuses on global variance (often losing local cluster structures). t-SNE is **non-linear** and specifically optimizes to keep similar neighbors close, preserving local clusters.
2.  **Q:** What is the 'Crowding Problem' in dimensionality reduction, and how does t-SNE solve it?
    *   **A:** In high dimensions, there is more volume available. When flattening to 2D, points get 'crowded' together. t-SNE uses a **t-distribution** (heavy-tailed) in the low-dimensional space to push non-neighbors further apart, decluttering the visualization.

### 🏢 Microsoft
3.  **Q:** Can I use t-SNE to reduce dimensions for a Machine Learning classifier?
    *   **A:** Generally **No**. t-SNE doesn't learn a parametric function (no `.transform()` method for new data). It is purely for exploring the training data. Use PCA or Autoencoders for feature pre-processing.
