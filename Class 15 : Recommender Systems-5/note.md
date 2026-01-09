# Notes: Recommender Systems - 5 (Class 15)

## Matrix Factorization & Related Algorithms

### 1. Unified Perspective
*   **Matrix Factorization (MF):** The general idea of approximating a matrix $A$ as the product of smaller matrices.
*   **K-Means as MF:** K-Means can be seen as a special case of MF where the "User Matrix" is a Cluster Assignment Matrix containing only binary values (0 or 1), with the constraint that each row sums to 1.
*   **GMM (Gaussian Mixture Models):** A "soft" clustering version where assignments are probabilities.
*   **NMF (Non-negative Matrix Factorization):** MF with the constraint that all factors must be non-negative ($\ge 0$). Useful for data like images (pixel intensities) or text counts where negative values don't make sense.

### 2. Singular Value Decomposition (SVD)
SVD is a fundamental linear algebra decomposition:
$$ A = U \cdot \Sigma \cdot V^T $$
*   **$U$ (Left Singular Vectors):** Maps Users to Latent Features.
*   **$\Sigma$ (Sigma):** Diagonal matrix of **Singular Values**. It represents the "strength" or "importance" of each latent feature.
*   **$V^T$ (Right Singular Vectors):** Maps Items to Latent Features.
*   **Analogy:** Like separating a music track (dataset) into individual instruments (latent patterns) and their volumes (sigma).
*   **Applications:** Image Compression, Noise Reduction, Topic Modelling, and Recommender Systems.

### 3. Tuning Hyperparameters
*   **Choosing 'd' (Number of Latent Factors):**
    *   **Trade-off:**
        *   Low $d$: High bias (underfitting), might miss subtle patterns.
        *   High $d$: High variance (overfitting), computationally expensive.
    *   **Elbow Method:** Plot **Loss vs. $d$**. As $d$ increases, loss decreases. The "elbow" point where the gain in performance diminishes is widely considered optimal.
*   **Validation Strategy:**
    *   **Split:** Divide known ratings into Train (e.g., 80%) and Validation (20%).
    *   **Train:** Run MF on the 80% data for various values of $d$.
    *   **Evaluate:** Check error (RMSE) on the held-out 20%.
    *   **Select:** Choose the $d$ that minimizes Validation Error to avoid overfitting.
