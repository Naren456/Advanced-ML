# 🎤 Real-World Interview Viva: GMM

### 🏢 Google / DeepMind
1.  **Q:** Explain the difference between Hard Clustering and Soft Clustering.
    *   **A:** K-Means is **Hard Clustering** (Point A belongs 100% to Cluster 1). GMM is **Soft Clustering** (Point A is 70% Cluster 1, 30% Cluster 2). This is useful for ambiguous data points.
2.  **Q:** How many parameters do you need to estimate for a GMM with $K$ components in $D$ dimensions?
    *   **A:** For each component: Mean ($D$), Covariance ($D 	imes D$ roughly), and Mixing Coefficient (scalar). It's much more complex than K-Means.
