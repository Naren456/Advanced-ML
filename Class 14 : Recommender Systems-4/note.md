# Class 14: Matrix Factorization for Recommender Systems

> **Core Principle:** "Decomposing interactions into latent user preferences and item characteristics."

---

## Table of Contents
1. [Introduction to Matrix Factorization](#1-introduction-to-matrix-factorization)
2. [Concept & Intuition](#2-concept--intuition)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Optimization Techniques](#4-optimization-techniques)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Introduction to Matrix Factorization

### 1.1 Context: The Netflix Prize
In 2006, Netflix launched a competition to improve their movie recommendation accuracy by 10%. The winning solution heavily relied on **Matrix Factorization (MF)** techniques, revolutionizing how we approach personalized recommendations.

### 1.2 Why Matrix Factorization?
- **Handling Sparsity:** Real-world rating matrices are 99% empty (sparse).
- **Latent Features:** It uncovers hidden patterns (like "dark comedy" or "80s action") that aren't explicitly tagged.
- **Scalability:** Efficiently handles millions of users and items.

---

## 2. Concept & Intuition

### 2.1 The Factorization Analogy
Consider the number **6**. We can "factorize" it into:
$$6 = 2 \times 3$$

Similarly, we can take a massive **User-Item Matrix (A)** and approximate it as the product of two smaller, dense matrices: **User Matrix (B)** and **Item Matrix (C)**.

$$A_{n \times m} \approx B_{n \times d} \times C_{d \times m}$$

Where:
- $n$: Number of Users
- $m$: Number of Items
- $d$: Number of **Latent Factors** (typically 10-100)

### 2.2 Intuition: Latent Dimensions
Imagine $d=2$. The hidden dimensions might represent:
1.  **Genre preference:** Romance vs. Action
2.  **Era:** Classic vs. Modern

**Example:**
- **User A:** Loves explosions, hates love stories. $\rightarrow$ Vector: `[+5 (Action), -3 (Romance)]`
- **Movie B (Die Hard):** High action, low romance. $\rightarrow$ Vector: `[+4 (Action), -1 (Romance)]`

**Predicted Rating:** Dot product of user and movie vectors.
$$R_{AB} \approx 5 \times 4 + (-3) \times (-1) = 20 + 3 = 23 \text{ (High score!)}$$

---

## 3. Mathematical Foundation

### 3.1 The Representation
We approximate the rating matrix $R$ using user factors $P$ and item factors $Q$:

$$\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i^T = \sum_{f=1}^{d} p_{uf} q_{if}$$

- $\mathbf{p}_u$: Preference vector for user $u$
- $\mathbf{q}_i$: Characteristic vector for item $i$

### 3.2 The Loss Function
Our goal is to minimize the difference between actual ratings ($r_{ui}$) and predicted ratings ($\hat{r}_{ui}$).

 **Objective:** Minimize Squared Error + **Regularization**

$$L = \sum_{(u,i) \in \text{Observed}} (r_{ui} - \mathbf{p}_u \cdot \mathbf{q}_i)^2 + \lambda (||\mathbf{p}_u||^2 + ||\mathbf{q}_i||^2)$$

**Why Regularization ($\lambda$)?**
To prevent overfitting. Without it, the model might learn unique noise for each user/item instead of generalizable patterns.

---

## 4. Optimization Techniques

How do we find the best values for $P$ and $Q$?

### 4.1 Stochastic Gradient Descent (SGD)
Update parameters iteratively for each individual rating.

**Algorithm:**
1. Initialize $P$ and $Q$ randomly.
2. For each known rating $r_{ui}$:
   - Calculate error: $e_{ui} = r_{ui} - \mathbf{p}_u \cdot \mathbf{q}_i$
   - Update User Vector: $\mathbf{p}_u \leftarrow \mathbf{p}_u + \gamma (e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u)$
   - Update Item Vector: $\mathbf{q}_i \leftarrow \mathbf{q}_i + \gamma (e_{ui} \mathbf{p}_u - \lambda \mathbf{q}_i)$
3. Repeat until convergence.

*$\gamma$: Learning rate*

**Step-by-Step Numerical Example:**
Suppose we want to predict user $u$'s rating for item $i$, where the actual rating $r_{ui} = 5$.
- Current User Vector: $\mathbf{p}_u = [0.1, 0.2]$
- Current Item Vector: $\mathbf{q}_i = [0.5, 0.3]$
- Learning Rate $\gamma = 0.01$, Regularization $\lambda = 0.02$

**Step 1: Predict Rating**
$$\hat{r}_{ui} = (0.1 \times 0.5) + (0.2 \times 0.3) = 0.05 + 0.06 = 0.11$$

**Step 2: Calculate Error**
$$e_{ui} = 5 - 0.11 = 4.89$$

**Step 3: Update User Vector ($\mathbf{p}_u$)**
$$\mathbf{p}_{u,new} = [0.1, 0.2] + 0.01 \times (4.89 \cdot [0.5, 0.3] - 0.02 \cdot [0.1, 0.2])$$
$$\mathbf{p}_{u,new} = [0.1, 0.2] + 0.01 \times ([2.445, 1.467] - [0.002, 0.004])$$
$$\mathbf{p}_{u,new} \approx [0.124, 0.215] \quad \text{(Values increased to match high rating)}$$

### 4.2 Alternating Least Squares (ALS)
Fix one matrix, solve for the other.

**Concept:**
- If $P$ is fixed, the problem becomes a quadratic (convex) problem for $Q$. We can solve exactly.
- **Step 1:** Fix $P$, solve for optimal $Q$.
- **Step 2:** Fix $Q$, solve for optimal $P$.
- **Repeat.**

**Example Walkthrough:**
Imagine a matrix with incomplete values:
$$R = \begin{bmatrix} 5 & ? \\ ? & 4 \end{bmatrix}, \quad P = \begin{bmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{bmatrix}, \quad Q = \begin{bmatrix} q_{11} & q_{12} \\ q_{21} & q_{22} \end{bmatrix}$$

1. **Initialize** $P$ randomly.
2. **Fix $P$:** Now, for the first row of $R$ (User 1), we want $P_1 \cdot Q^T \approx [5, ?]$. This becomes a standard **Least Squares** problem (like simple regression) to find the best $Q$ that fits the known rating (5).
3. **Solve for $Q$:** We get an exact optimal $Q$ for the current $P$.
4. **Fix $Q$:** Now treat $Q$ as constant and solve for the best $P$ to fit the validation data.
5. **Repeat** until the error stops decreasing.

**Why ALS?**
- **Parallelizable:** Calculations for each user/item are independent.
- **Implicit Feedback:** works well when we only have "clicks" or "views" rather than explicit star ratings.


**Step-by-Step Numerical Example:**

**Step 1: Original Matrix ($R$)**

Suppose we have a user–item rating matrix equal to $R$.
*   2 users, 2 items
*   `?` means **missing rating**
*   We choose **rank ($k = 1$)** (one latent factor)

$$
R = \begin{bmatrix} 5 & 3 \\ 4 & ? \end{bmatrix}
$$

**Step 2: Factor Matrices**

We want $R \approx U V^T$. Since $k=1$, all values are scalars.
$$
U = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}, \quad V = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}
$$

**Step 3: Initialize ($U$)**

Start with random values (simple ones for clarity):
$$
U = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

**Step 4: Fix ($U$), solve for ($V$)**

*   **Item 1:**
    *   Ratings: User 1 rated 5, User 2 rated 4.
    *   Minimize: $(5 - u_1 v_1)^2 + (4 - u_2 v_1)^2$
    *   Substitute $u_1 = u_2 = 1$: $(5 - v_1)^2 + (4 - v_1)^2$
    *   Derivative: $-2(5 - v_1) - 2(4 - v_1) = 0 \Rightarrow 2(2v_1 - 9) = 0 \Rightarrow v_1 = 4.5$

*   **Item 2:**
    *   Ratings: User 1 rated 3.
    *   Minimize: $(3 - u_1 v_2)^2 = (3 - v_2)^2$
    *   Solution: $v_2 = 3$

*   **Updated ($V$):**
    $$V = \begin{bmatrix} 4.5 \\ 3 \end{bmatrix}$$

**Step 5: Fix ($V$), solve for ($U$)**

*   **User 1:**
    *   Ratings: Item 1 $\rightarrow$ 5, Item 2 $\rightarrow$ 3.
    *   Minimize: $(5 - u_1 \cdot 4.5)^2 + (3 - u_1 \cdot 3)^2$
    *   Derivative: $-2 \cdot 4.5(5 - 4.5u_1) - 2 \cdot 3(3 - 3u_1) = 0$
    *   $40.5u_1 - 45 + 18u_1 - 18 = 0 \Rightarrow 58.5u_1 = 63 \Rightarrow u_1 \approx 1.08$

*   **User 2:**
    *   Ratings: Item 1 $\rightarrow$ 4.
    *   Minimize: $(4 - u_2 \cdot 4.5)^2 \Rightarrow u_2 = \frac{4}{4.5} \approx 0.89$

*   **Updated ($U$):**
    $$U = \begin{bmatrix} 1.08 \\ 0.89 \end{bmatrix}$$

**Step 6: Reconstructed Matrix**

$$
\hat{R} = U V^T = \begin{bmatrix} 1.08 \\ 0.89 \end{bmatrix} \begin{bmatrix} 4.5 & 3 \end{bmatrix} = \begin{bmatrix} 4.86 & 3.24 \\ 4.00 & 2.67 \end{bmatrix}
$$

**Predicted missing value:**
$$ \boxed{R_{2,2} \approx 2.67} $$

**Key Takeaway:**
*   ALS **alternates** between fixing $U$ and $V$.
*   Each step solves a **least squares problem**.
*   Even with missing data, ALS can predict unknown entries.

### 4.3 SGD vs. ALS

| Feature | SGD | ALS |
| :--- | :--- | :--- |
| **Speed** | Faster per iteration | Slower per iteration |
| **Scalability** | Good, but sequential | Excellent (Massive Parallelization) |
| **Data Type** | Best for explicit ratings | Best for implicit feedback |
| **Math** | Gradient-based Approx. | Exact Least Squares solution per step |

---

## 5. Implementation

### 5.1 Basic Python Structure (Conceptual)

```python
import numpy as np

class MatrixFactorization:
    def __init__(self, R, K=20, alpha=0.01, beta=0.02, steps=1000):
        self.R = R          # User-Item Matrix
        self.K = K          # Latent factors (d)
        self.alpha = alpha  # Learning rate
        self.beta = beta    # Regularization (lambda)
        self.steps = steps  # Iterations

    def train(self):
        N, M = self.R.shape
        self.P = np.random.rand(N, self.K)
        self.Q = np.random.rand(M, self.K)

        for step in range(self.steps):
            for i in range(N):
                for j in range(M):
                    if self.R[i][j] > 0:
                        eij = self.R[i][j] - np.dot(self.P[i,:], self.Q[j,:].T)
                        # SGD Updates
                        self.P[i,:] += self.alpha * (2 * eij * self.Q[j,:] - self.beta * self.P[i,:])
                        self.Q[j,:] += self.alpha * (2 * eij * self.P[i,:] - self.beta * self.Q[j,:])
        return self.P, self.Q

# Usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

mf = MatrixFactorization(R, K=2)
P, Q = mf.train()
approximation = np.dot(P, Q.T)
print(approximation)
```

### 5.2 Industry Standard: `Surprise` Library
In production, use optimized libraries:

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# 1. Load Data
data = Dataset.load_builtin('ml-100k')

# 2. Algo (SVD is essentially Matrix Factorization discussed here)
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

# 3. Validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

---

## 6. Exam Preparation

### 6.1 Key Formulas Summary
| Concept | Formula |
| :--- | :--- |
| **Prediction** | $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$ (Including biases) |
| **Loss Function** | $L = \sum (r_{ui} - \hat{r}_{ui})^2 + \lambda||\theta||^2$ |
| **Sparsity** | $1 - \frac{\text{Non-zero entries}}{\text{Total entries}}$ |

### 6.2 Interview Questions

**Q1: What is the "Cold Start" problem in Matrix Factorization?**
**A:** MF relies on past interactions to learn vectors $\mathbf{p}_u$ and $\mathbf{q}_i$. New users/items have no interactions, so their vectors are random/zero.
*Solution:* Hybrid methods (content-based), showing popular items to new users.

**Q2: Why use Regularization $(\lambda)$?**
**A:** To penalize large weights. If a user only watched 1 movie, we don't want their vector to shift drastically to match that single rating perfectly. It keeps the model "simple" and generalizable.

**Q3: Explain the difference between SVD and Matrix Factorization for Recommenders.**
**A:**
- **Standard SVD:** Defined for complete matrices. Requires filling missing values (imputation), which introduces bias and is computationally expensive ($O(N^3)$).
- **MF (RecSys):** Only models observed ratings. Ignores missing values in the loss function. Uses optimization (SGD/ALS) instead of analytical algebra.

**Q4: How does ALS enable parallelization?**
**A:** When $P$ is fixed, the optimal $Q$ for each item $i$ depends *only* on the users who rated item $i$. Thus, we can compute every item vector independently (in parallel). Same for users when $Q$ is fixed.

---
**Next Class:** We will explore **Hybrid Systems** and Deep Learning approaches to overcome the limitations of standard Matrix Factorization.
