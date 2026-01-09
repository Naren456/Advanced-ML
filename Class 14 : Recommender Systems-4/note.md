# Notes: Recommender Systems - 4 (Class 14)

## Matrix Factorization (MF) Breakdown

### 1. The Core Idea
*   **Problem:** The User-Item Matrix ($A$) is huge and sparse (mostly empty). Using standard methods is inefficient.
*   **Solution:** Decompose/Factorize the matrix $A$ ($n \times m$) into two lower-dimensional matrices:
    *   **User Matrix ($B$):** $n \times d$ (Users $\times$ Latent Factors)
    *   **Item Matrix ($C$):** $d \times m$ (Latent Factors $\times$ Items)
*   **Latent Factors ($d$):** Hidden features that explain preferences (e.g., "Action vs Romance", "Old vs New"). The algorithm learns these automatically.
*   **Prediction:** The predicted rating for User $i$ and Item $j$ is the dot product of their vectors:
    $$ \hat{r}_{ij} = B_i \cdot C_j $$

### 2. Learning the Factors
*   We learn the values in $B$ and $C$ by minimizing the difference between the **Predicted Rating** and the **Actual Rating** (for the few ratings we actually have).
*   **Loss Function:**
    $$ L = \sum (r_{ij} - B_i \cdot C_j)^2 + \lambda (||B_i||^2 + ||C_j||^2) $$
    *   The first term is the **Squared Error**.
    *   The second term is **Regularization** ($\lambda$) to prevent overfitting (keeps values small).

### 3. Optimization Algorithms
How do we find the best $B$ and $C$ to minimize Loss?

#### A. Stochastic Gradient Descent (SGD)
*   **Method:**
    1.  Initialize $B$ and $C$ randomly.
    2.  For each known rating, calculate the error.
    3.  Update the user and item vectors slightly in the opposite direction of the gradient.
    4.  Repeat until convergence.
*   **Pros:** Simple, accurate.
*   **Cons:** Can be slow on huge datasets; sensitive to learning rate.

#### B. Alternating Least Squares (ALS)
*   **Method:**
    1.  Fix Matrix $B$ (treat it as constant) $\rightarrow$ The problem becomes a simple quadratic (easy to solve) to find best $C$.
    2.  Fix Matrix $C$ (treat it as constant) $\rightarrow$ Solve for best $B$.
    3.  Alternate back and forth until convergence.
*   **Pros:** Handles massive datasets well (parallelizable), very stable.
*   **Cons:** Mathematically slightly more complex per step.

### 4. Connection to SVD
*   **SVD (Singular Value Decomposition):** A linear algebra technique for dimensionality reduction. Matrix Factorization in RecSys is essentially a functional approximation of SVD optimized for sparse data.
