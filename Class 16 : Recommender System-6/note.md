# Notes: Recommender Systems - 6 (Class 16)

## Matrix Factorization Implementation (From Scratch)

### 1. Data Setup
*   **Datasets:** `movies.csv`, `ratings.csv`, `users.csv` (likely MovieLens dataset).
*   **Interaction Matrix ($R$):**
    *   Created by pivoting the `ratings` DataFrame.
    *   Index columns: `userId`.
    *   Columns: `movieId`.
    *   Values: `rating`.
    *   **Handling Missing Values:** Filled with $0$. Ideally, we should only train on known ratings, but filling with 0 is a common simplification for basic implementations (treating them as explicit "not liked" or just placeholders).

### 2. The Algorithm: Stochastic Gradient Descent (SGD)
The goal is to decompose $R$ into User Matrix $P$ and Item Matrix $Q^T$ such that $R \approx P \times Q^T$.

*   **Initialization:**
    *   $K$: Number of latent factors (e.g., 2).
    *   $P$ (Users $\times K$) and $Q$ (Items $\times K$): Initialized with random normal values.
    *   Hyperparameters: `steps` (epochs), `alpha` (learning rate), `beta` (regularization strength).

*   **Training Loop:**
    1.  Iterate for a fixed number of `steps`.
    2.  Loop through every user $i$ and item $j$.
    3.  **Filter:** Only process if $R_{ij} > 0$ (observed ratings).
    4.  **Prediction:** $\hat{r}_{ij} = P_i \cdot Q_j^T$.
    5.  **Error:** $e_{ij} = R_{ij} - \hat{r}_{ij}$.
    6.  **Update Weights (Gradient Descent with Regularization):**
        *   $P_{ik}^{new} = P_{ik} + \alpha \cdot (2e_{ij}Q_{jk} - \beta P_{ik})$
        *   $Q_{jk}^{new} = Q_{jk} + \alpha \cdot (2e_{ij}P_{ik} - \beta Q_{jk})$

### 3. Prediction
*   After training, the full predicted matrix is obtained by $\hat{R} = P \times Q^T$.
*   This fills in the zeros (missing values) with predicted ratings, which can be used to recommend top items to users.

### 4. Code Snippet Concepts
```python
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0: # Only train on known ratings
                    eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        # Gradient Descent Updates
                        P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    return P, Q.T
```
