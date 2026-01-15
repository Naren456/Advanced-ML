# Class 15: The Unified View of Matrix Factorization

> **Core Principle:** "Matrix Factorization isn't just for recommender systems; it's the hidden engine behind K-Means, PCA, SVD, and many other unsupervised learning algorithms."

---

## Table of Contents
1. [Logistics & Project Updates](#1-logistics--project-updates)
2. [The Grand Unification: Framework](#2-the-grand-unification-framework)
3. [Algorithm 1: K-Means as Matrix Factorization](#3-algorithm-1-k-means-as-matrix-factorization)
4. [Algorithm 2: Gaussian Mixture Models (GMM)](#4-algorithm-2-gaussian-mixture-models-gmm)
5. [Algorithm 3: Recommender Systems](#5-algorithm-3-recommender-systems)
6. [Algorithm 4: Non-Negative Matrix Factorization (NMF)](#6-algorithm-4-non-negative-matrix-factorization-nmf)
7. [Algorithm 5: Singular Value Decomposition (SVD)](#7-algorithm-5-singular-value-decomposition-svd)
8. [Algorithm 6: Principal Component Analysis (PCA)](#8-algorithm-6-principal-component-analysis-pca)
9. [Exam Preparation](#9-exam-preparation)

---

## 1. Logistics & Project Updates

### 1.1 Project Overview
We are launching a comprehensive project focusing on **Unsupervised Learning Algorithms**.
- **Scope:** End-to-End Data Analysis.
- **Components:**
    - Exploratory Data Analysis (EDA)
    - Feature Engineering
    - Model Building (Clustering, Dimensions Reduction, Recommenders)
    - Evaluation
- **Goal:** To see how these algorithms perform on real-world data variants.

### 1.2 Schedule
- **Attendance:** Updates expected by EOD.
- **Cancellations:** Potential scheduling adjustments for upcoming Friday/Monday.
- **Tip:** Keep concepts in your "cache memory" for the project work!

---

## 2. The Grand Unification: Framework

In Machine Learning, specifically Unsupervised Learning, we often deal with a large data matrix $X$. The goal is almost always to **approximate** this large matrix as the product of smaller, meaningful matrices.

$$
X \approx A \times B^T
$$

Depending on the **constraints** we place on $A$ and $B$, we get different algorithms.

---

## 3. Algorithm 1: K-Means as Matrix Factorization

We typically think of K-Means as "finding centroids." But mathematically, it's a matrix factorization problem.

### 3.1 The Equation
$$ X \approx Z \cdot C $$

Where:
- $X$: Original Data Matrix ($N \times M$) (Points $\times$ Features)
- $Z$: **Cluster Assignment Matrix** ($N \times K$)
- $C$: **Centroids Matrix** ($K \times M$)

### 3.2 The Constraints
1.  **Binary Assignment:** $Z_{ij} \in \{0, 1\}$. A point belongs to a cluster or it doesn't.
2.  **Row Sum:** $\sum_j Z_{ij} = 1$. Each point must belong to exactly **one** cluster.

### 3.3 The "Solving Matrix" Example
Let's see the numbers in action. Suppose we have 3 data points ($N=3$) and 2 features ($M=2$):

$$
X = \begin{bmatrix} 
1.2 & 1.0 \\ 
1.0 & 1.2 \\ 
5.1 & 4.9 
\end{bmatrix}
$$

We want $K=2$ clusters. 

**Step 1: The Centroids Matrix ($C$)**  
Suppose closer centroids are found at roughly $(1.1, 1.1)$ and $(5.0, 5.0)$.
$$
C_{centroids} = \begin{bmatrix} 
1.1 & 1.1 \\ 
5.0 & 5.0 
\end{bmatrix}
$$

**Step 2: The Assignment Matrix ($Z$)**  
K-Means assigns each point to the closest centroid.
- Points 1 & 2 are close to Cluster 1.
- Point 3 is close to Cluster 2.
$$
Z_{assignment} = \begin{bmatrix} 
1 & 0 \\ 
1 & 0 \\ 
0 & 1 
\end{bmatrix}
$$

**Step 3: Matrix Multiplication ($Z \times C$)**  
When we multiply these, we get the **Reconstructed Matrix** ($X'$), which represents the "simplest version" of our data where every point is exactly at its centroid.

$$
X' \approx Z \times C = \begin{bmatrix} 
1 & 0 \\ 
1 & 0 \\ 
0 & 1 
\end{bmatrix} \times \begin{bmatrix} 
1.1 & 1.1 \\ 
5.0 & 5.0 
\end{bmatrix} = \begin{bmatrix} 
1.1 & 1.1 \\ 
1.1 & 1.1 \\ 
5.0 & 5.0 
\end{bmatrix}
$$

**Conclusion:** K-Means minimizes the difference between the original $X$ and this reconstructed $X'$.
$$ \text{Loss} = || X - Z \cdot C ||^2 $$

### 3.4 Code: K-Means as Transformation
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1.2, 1.0], [1.0, 1.2], [5.1, 4.9]])

# Fit K-Means
kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X) # Added random_state and n_init for reproducibility

# The Matrices
Z = kmeans.predict(X)          # Assignment (Conceptually)
C = kmeans.cluster_centers_    # Centroids Matrix
print(f"Centroids (C):\n{C}")
```

### 3.5 Intuition
If you multiply the assignment matrix $Z$ by the centroids $C^T$, you reconstruct the original data points by replacing each point with the coordinate of its assigned centroid. K-Means tries to minimize the error of this reconstruction (Inertia).

---

## 4. Algorithm 2: Gaussian Mixture Models (GMM)

GMM is the "Soft" cousin of K-Means.

### 4.1 The Equation
Similar to K-Means:
$$ X \approx Z \cdot C $$

### 4.2 The Constraints
- **Probabilistic Assignment:** $Z_{ij} \in [0, 1]$.
- **Constraint:** values in $Z$ represent the *probability* ($P(Cluster_j | Point_i)$) that a point belongs to a cluster.

### 4.3 Detailed Matrix Example
Unlike K-Means where $Z$ is 0 or 1, here $Z$ is "Soft".

$$
Z_{GMM} = \begin{bmatrix} 
0.9 & 0.1 \\ 
0.8 & 0.2 \\ 
0.4 & 0.6 
\end{bmatrix}
$$

Row 1 says: "I am 90% sure this point belongs to Cluster 1".  
Row 3 says: "I am uncertain, maybe 40% Cluster 1, 60% Cluster 2".

When you multiply $Z \times C$, the reconstructed point is a **weighted average** of all centroids, pulling it slightly towards everyone.

### 4.4 Code: GMM Soft Clustering
```python
from sklearn.mixture import GaussianMixture
import numpy as np # Ensure numpy is imported for X

X = np.array([[1.2, 1.0], [1.0, 1.2], [5.1, 4.9]]) # Re-define X for GMM example

gmm = GaussianMixture(n_components=2, random_state=0).fit(X) # Added random_state for reproducibility

# Get the Soft Assignment Matrix (Z)
Z_probs = gmm.predict_proba(X)
print(f"Soft Assignments (Z):\n{Z_probs}")
```

*Note: In GMM, we also learn Covariance, but the core "location" logic follows this factorization structure.*

---

## 5. Algorithm 3: Recommender Systems

This is our primary focus for the module.

### 5.1 The Equation
$$ R_{N \times M} \approx U_{N \times D} \times V^T_{D \times M} $$

Where:
- $N$: Users
- $M$: Items
- $D$: **Latent Factors** (Hyperparameter)

### 5.2 Choosing 'D' (Number of Latent Factors)
How do we decide if $D$ should be 5, 20, or 100?

#### Method A: The Elbow Curve
1.  Train model with $D=1, 2, \dots, 50$.
2.  Calculate **Reconstruction Error (RMSE)** for each.
3.  Plot **Error vs. $D$**.
4.  Pick the point where the drop in error "flattens out" (diminishing returns).

#### Method B: Train-Validation Split (Preferred)
1.  **Split Data:** 80% Training (mask 20% ratings), 20% Validation.
2.  Train models with different $D$ on the 80% set.
3.  Evaluate error on the hidden 20% set.
4.  **Select $D$** that minimizes **Validation Error**.

### 5.3 Working Through the Matrix (The "Solving")
Imagine we have ratings for 4 Users and 3 Movies ($4 \times 3$). We pick $D=2$ latent factors ("Action-ness", "Comedy-ness").

$$
\text{Ratings } (R) \approx \text{Users } (U) \times \text{Items } (V^T)
$$

$$
\begin{bmatrix} 
5 & 3 & - \\ 
- & 4 & 5 \\ 
1 & 1 & - \\ 
- & - & 5 
\end{bmatrix}
\approx
\underbrace{\begin{bmatrix} 
1.2 & 0.2 \\ 
1.1 & 1.5 \\ 
0.1 & -0.5 \\ 
1.5 & 1.8 
\end{bmatrix}}_{\text{User Preferences}}
\times
\underbrace{\begin{bmatrix} 
2.0 & 1.0 & 0.1 \\ 
0.5 & 3.0 & 2.5 
\end{bmatrix}}_{\text{Movie Genres}}
$$

**Let's predict the rating for User 1 on Movie 2:**
1. **User 1 Vector:** $[1.2, 0.2]$ (Likes Feature 1 "Action", indifferent to Feature 2 "Comedy").
2. **Movie 2 Vector:** $[1.0, 3.0]^T$ (Mild Action, High Comedy).
3. **Dot Product:**
   $$ \text{Pred} = (1.2 \times 1.0) + (0.2 \times 3.0) = 1.2 + 0.6 = 1.8 $$
   
   *Result: User 1 probably won't like Movie 2 that much (Rating 1.8), because they strictly prefer Action over Comedy.*

### 5.4 Finding the Best Matrices (ALS/SGD)
The computer starts with random numbers for $U$ and $V$, then keeps tweaking them to minimize the error on the ratings we **do** know (5, 3, 4, etc.).

### 5.5 Interpreting 'D'
- **Visualizing:** You can use **t-SNE** to project the $D$-dimensional vectors into 2D to see clusters of similar users or items.
- **Meaning:** In practice, latent factors don't always map to clear concepts like "Horror" or "Comedy". They are abstract mathematical features that best explain the variance in ratings.

---

## 6. Algorithm 4: Non-Negative Matrix Factorization (NMF)

### 6.1 The Definition
A Matrix Factorization where **all matrices must be non-negative** ($ \ge 0$).

$$ X \approx W \times H, \quad \text{where } W \ge 0, H \ge 0 $$

### 6.2 Applications
- **Image Processing:** Pixels are values $0-255$ (never negative). NMF decomposes a face into "parts" (nose, eyes, mouth) rather than abstract "eigenfaces".
- **Cluster Interpretation:** Since values are additive (no negative cancellation), the components are often more interpretable parts of the whole.

### 6.3 Code: NMF for Topic Modeling (Example)
```python
from sklearn.decomposition import NMF

# Imagine X is a Term-Frequency Matrix (Documents x Words)
# X = ... 

nmf = NMF(n_components=5, init='random', random_state=0).fit(X)

# W: Document-Topic Matrix
W = nmf.transform(X)

# H: Topic-Word Matrix
H = nmf.components_
print(f"Topic-Word Matrix Shape: {H.shape}")
```

---

## 7. Algorithm 5: Singular Value Decomposition (SVD)

The "King" of Matrix Factorization in Linear Algebra.

### 7.1 The Formula
$$ X = U \cdot \Sigma \cdot V^T $$

- **$U$ (Left Singular Vectors):** Relationship between Row entities (e.g., Users) and Concepts.
- **$\Sigma$ (Sigma):** Diagonal matrix of **Singular Values**. Represents the "Importance" or "Strength" of each concept.
- **$V^T$ (Right Singular Vectors):** Relationship between Concepts and Column entities (e.g., Items).

### 7.2 Music Studio Analogy
Think of a song ($X$):
- **$U$:** The sheet music (Notes/Patterns).
- **$\Sigma$:** The Mixing Board (Volume sliders for each instrument).
- **$V^T$:** The actual Instruments.

### 7.3 Applications
1.  **Image Compression:** Keep only the top $k$ singular values (largest $\Sigma$). You reconstruct the image with far less data.
2.  **Noise Reduction:** Small singular values often represent noise. setting them to 0 removes static.
3.  **Topic Modeling (LSA):** Document-Term Matrix factorization.

### 7.4 Code: SVD (TruncatedSVD in sklearn)
```python
from sklearn.decomposition import TruncatedSVD

# X is our data matrix
svd = TruncatedSVD(n_components=2, random_state=0).fit(X)

# U * Sigma (approx)
X_reduced = svd.transform(X)

print(f"Explained Variance: {svd.explained_variance_ratio_}")
```

---

## 8. Algorithm 6: Principal Component Analysis (PCA)

### 8.1 The Setup
PCA is actually a specific application of SVD focused on **Covariance**.

$$ S = W \cdot \Lambda \cdot W^T $$

Where:
- $S$: Covariance Matrix of Data ($X^T X$)
- $W$: Eigenvectors (Principal Components)
- $\Lambda$: Eigenvalues (Variance explained)

### 8.2 Homework
**Task:** Prove/derive how PCA's objective function relates to the Matrix Factorization formulation.
*Hint: PCA minimizes reconstruction error squared, just like Matrix Factorization.*

---

## 9. Exam Preparation

### 9.1 Key Relationships Table

| Algorithm | Matrix Form | Constraint 1 | Constraint 2 |
| :--- | :--- | :--- | :--- |
| **K-Means** | $X \approx Z \cdot C$ | $Z$ is Binary $\{0,1\}$ | Row Sum of $Z = 1$ |
| **GMM** | $X \approx Z \cdot C$ | $Z \in [0,1]$ (Prob) | Row Sum of $Z = 1$ |
| **RecSys** | $R \approx U \cdot V^T$ | None (usually) | Regularization on Norms |
| **NMF** | $X \approx W \cdot H$ | $W, H \ge 0$ | - |
| **SVD** | $X = U \Sigma V^T$ | $U, V$ are Orthogonal | $\Sigma$ is Diagonal |

### 9.2 Interview Questions

**Q1: How is K-Means related to Matrix Factorization?**
**A:** K-Means is a hard-assignment matrix factorization where the user matrix $Z$ is constrained to binary values indicating cluster membership, and $C$ represents centroids.

**Q2: Why use Validation Split to tune 'D' instead of Training Error?**
**A:** As $D$ increases, Training Error *always* decreases (overfitting). Validation error will decrease and then start increasing when the model starts fitting noise. We want the $D$ at that inflection point.

**Q3: What is the physical meaning of negative values in SVD?**
**A:** In SVD/PCA, negative values allow for "correction" or "subtraction" from the mean. If a "Face" component adds a beard, a negative weight might remove it. NMF forbids this, forcing "additive only" parts.

---
**Next Class:** We will move from theory to code, implementing Matrix Factorization from scratch and using SVD for Recommender systems.
