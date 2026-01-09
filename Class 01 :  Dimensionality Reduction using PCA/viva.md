# 🎤 Real-World Interview Viva: PCA

### 🏢 Amazon / Google (Concepts)
1.  **Q:** Explain the "Curse of Dimensionality" to a non-technical stakeholder.
    *   **A:** "Imagine finding a needle. In a 1-meter haystack, it's hard. In a haystack the size of the universe (high dimensions), it's impossible. Adding more features makes our data mostly 'empty space', confusing the model."
2.  **Q:** Does PCA work for non-linear data? How would you handle it?
    *   **A:** No, PCA finds linear projections. For non-linear data (like a Swiss Roll), use **Kernel PCA**, **t-SNE**, or **Autoencoders**.
3.  **Q:** Why do we examine the Cumulative Explained Variance ratio?
    *   **A:** To determine the optimal number of components ($k$). We typically stop when we retain 90-95% of information to trade off complexity vs. accuracy.

### 🏢 Uber / Meta (Math & Application)
4.  **Q:** Mathematically, what is the first Principal Component?
    *   **A:** It is the eigenvector corresponding to the **largest eigenvalue** of the Covariance Matrix. It points in the direction of maximum variance.
5.  **Q:** You have a dataset where one feature is in 'meters' (range 0-1) and another in 'microns' (range 0-1,000,000). What happens if you run PCA without scaling?
    *   **A:** The 'microns' feature will dominate the first principal component solely because of its large variance/magnitude, not because it's more informative. **Standardization is mandatory.**
