# Class 13: Recommender Systems - Matrix Factorization Fundamentals

> **Core Principle:** "Decomposing sparse matrices to discover latent user-item relationships"

---

## Table of Contents
1. [Introduction to Matrix Factorization](#1-introduction)
2. [Mathematics of Factorization](#2-mathematics)
3. [Implementation](#3-implementation)
4. [Exam Preparation](#4-exam-preparation)

---

## 1. Introduction

### 1.1 The Problem with Collaborative Filtering

**User-Item Matrix is Huge and Sparse:**
```
         Item1  Item2  Item3  ... Item10K
User1      5      ?      3    ...    ?
User2      ?      4      ?    ...    1
...
User1M     ?      ?      5    ...    ?

99% of entries are missing!
```

**Scalability Issues:**
- User-User CF: $O(U^2)$ similarity computations
- Item-Item CF: $O(I^2)$ similarity computations
- Space: $O(U \times I)$ to store matrix

### 1.2 Matrix Factorization Solution

**Key Idea:** Decompose into smaller dense matrices

$$R_{n \times m} \approx P_{n \times k} \times Q_{k \times m}^T$$

**Benefits:**
1. **Compression:** Store $n \times k + m \times k$ instead of $n \times m$
2. **Generalization:** Learn patterns, not memorize ratings
3. **Predictions:** Compute $\hat{r}_{ij} = p_i \cdot q_j$ for any user-item pair

### 1.3 Latent Factors Intuition

**Example - Movies:**

Imagine $k=2$ latent dimensions represent:
- Factor 1: "Action vs. Drama"
- Factor 2: "Mainstream vs. Indie"

**User vector:** $p_{\text{Alice}} = [0.9, 0.2]$ → Likes action, mainstream
**Movie vector:** $q_{\text{Avengers}} = [0.95, 0.1]$ → Action, mainstream

**Predicted rating:**
$$\hat{r}_{\text{Alice,Avengers}} = 0.9 \times 0.95 + 0.2 \times 0.1 = 0.855 + 0.02 = 0.875$$

High score → Likely to enjoy!

---

## 2. Mathematics

### 2.1 Objective Function

**Goal:** Minimize reconstruction error

$$\min_{P,Q} \mathcal{L} = \sum_{(i,j) \in \Omega} (r_{ij} - p_i \cdot q_j)^2 + \lambda(\|P\|_F^2 + \|Q\|_F^2)$$

where:
- $\Omega$: Set of observed ratings
- $r_{ij}$: Actual rating
- $p_i \cdot q_j$: Predicted rating
- $\lambda$: Regularization parameter
- $\|\cdot\|_F$: Frobenius norm

**Regularization Term:** Prevents overfitting by penalizing large weights

### 2.2 Why $k$ Latent Factors?

**Trade-off:**
- **Small $k$ (5-10):** 
  - Fast, less overfitting
  - May underfit (can't capture complexity)
- **Large $k$ (100+):**
  - More expressive
  - Slower, more overfitting risk
  
**Typical range:** 10-50 factors

### 2.3 Matrix Dimensions

**Original:**
- Users: $n = 1{,}000{,}000$
- Items: $m = 10{,}000$
- Matrix size: $10^{10}$ entries (10 billion!)
- Sparsity: 99.9% empty

**Factorized ($k=20$):**
- $P$: $1{,}000{,}000 \times 20 = 20$ million
- $Q$: $10{,}000 \times 20 = 200$ thousand
- Total: ~20.2 million (500× reduction!)

---

## 3. Implementation

### 3.1 Basic Matrix Factorization

```python
import numpy as np

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=10, learning_rate=0.01,
                 reg_param=0.01, n_epochs=20, verbose=True):
        """
        Initialize MF model
        
        Parameters:
        -----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors
        learning_rate : float
            Learning rate for SGD
        reg_param : float
            L2 regularization parameter
        n_epochs : int
            Number of training epochs
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg_param
        self.n_epochs = n_epochs
        self.verbose = verbose
        
        # Initialize latent factor matrices randomly
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        
        # Track training history
        self.train_loss = []
    
    def fit(self, ratings):
        """
        Train the model using Stochastic Gradient Descent
        
        Parameters:
        -----------
        ratings : list of tuples
            List of (user_id, item_id, rating) tuples
        """
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)  # Shuffle for each epoch
            epoch_loss = 0
            
            for user, item, rating in ratings:
                # Prediction
                prediction = self.P[user] @ self.Q[item]
                
                # Error
                error = rating - prediction
                epoch_loss += error ** 2
                
                # Gradient descent updates
                p_grad = -2 * error * self.Q[item] + 2 * self.reg * self.P[user]
                q_grad = -2 * error * self.P[user] + 2 * self.reg * self.Q[item]
                
                self.P[user] -= self.lr * p_grad
                self.Q[item] -= self.lr * q_grad
            
            # Add regularization to loss
            reg_loss = self.reg * (np.sum(self.P**2) + np.sum(self.Q**2))
            total_loss = epoch_loss + reg_loss
            self.train_loss.append(total_loss)
            
            if self.verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
        
        return self
    
    def predict(self, user, item):
        """Predict rating for user-item pair"""
        return self.P[user] @ self.Q[item]
    
    def recommend(self, user, n=5, exclude=None):
        """
        Recommend top n items for user
        
        Parameters:
        -----------
        user : int
            User ID
        n : int
            Number of recommendations
        exclude : set
            Items to exclude (already rated)
        """
        # Score all items
        scores = self.P[user] @ self.Q.T
        
        # Exclude items
        if exclude:
            scores[list(exclude)] = -np.inf
        
        # Top N
        top_items = np.argsort(scores)[::-1][:n]
        top_scores = scores[top_items]
        
        return list(zip(top_items, top_scores))

# Example usage
# Create sample data
ratings_data = [
    (0, 0, 5.0), (0, 1, 3.0), (0, 3, 1.0),
    (1, 0, 4.0), (1, 3, 1.0),
    (2, 1, 1.0), (2, 2, 5.0), (2, 3, 4.0),
    (3, 2, 4.0), (3, 3, 5.0),
]

# Initialize and train
mf = MatrixFactorization(n_users=4, n_items=4, n_factors=2, 
                         learning_rate=0.01, reg_param=0.1, n_epochs=100)
mf.fit(ratings_data)

# Get predictions
print("\nPredictions:")
print(f"User 0, Item 2: {mf.predict(0, 2):.2f}")
print(f"User 1, Item 1: {mf.predict(1, 1):.2f}")

# Get recommendations
print("\nRecommendations for User 0:")
recs = mf.recommend(user=0, n=3, exclude={0, 1, 3})
for item, score in recs:
    print(f"  Item {item}: {score:.2f}")
```

### 3.2 Visualization

```python
import matplotlib.pyplot as plt

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(mf.train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True, alpha=0.3)
plt.show()

# Visualize latent factors (if k=2)
if mf.n_factors == 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # User factors
    ax1.scatter(mf.P[:, 0], mf.P[:, 1], s=100, alpha=0.6)
    for i in range(mf.n_users):
        ax1.annotate(f'U{i}', (mf.P[i, 0], mf.P[i, 1]))
    ax1.set_xlabel('Factor 1')
    ax1.set_ylabel('Factor 2')
    ax1.set_title('User Latent Factors')
    ax1.grid(True, alpha=0.3)
    
    # Item factors
    ax2.scatter(mf.Q[:, 0], mf.Q[:, 1], s=100, alpha=0.6, color='orange')
    for i in range(mf.n_items):
        ax2.annotate(f'I{i}', (mf.Q[i, 0], mf.Q[i, 1]))
    ax2.set_xlabel('Factor 1')
    ax2.set_ylabel('Factor 2')
    ax2.set_title('Item Latent Factors')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

---

## 4. Exam Preparation

### 4.1 Key Concepts

**Matrix Factorization vs Collaborative Filtering:**

| Aspect | CF (Similarity-based) | Matrix Factorization |
|--------|----------------------|---------------------|
| **Approach** | Find similar users/items | Learn latent factors |
| **Scalability** | $O(U^2)$ or $O(I^2)$ | $O(k(U+I))$ |
| **Sparsity** | Struggles | Handles well |
| **Interpretability** | High | Low |
| **Accuracy** | Good | Better |
| **Cold-start** | Difficult | Slightly better |

### 4.2 Common Exam Questions

**Q1: Derive the gradient for matrix factorization.**

**Solution:**

Loss for single rating:
$$\mathcal{L}_{ij} = (r_{ij} - p_i^T q_j)^2 + \lambda(\|p_i\|^2 + \|q_j\|^2)$$

Gradient w.r.t. $p_i$:
$$\frac{\partial \mathcal{L}}{\partial p_i} = \frac{\partial}{\partial p_i}\left[(r_{ij} - p_i^T q_j)^2 + \lambda\|p_i\|^2\right]$$

Using chain rule:
$$= -2(r_{ij} - p_i^T q_j) \cdot q_j + 2\lambda p_i$$

$$= -2e_{ij} \cdot q_j + 2\lambda p_i$$

Similarly for $q_j$:
$$\frac{\partial \mathcal{L}}{\partial q_j} = -2e_{ij} \cdot p_i + 2\lambda q_j$$

**Q2: How does regularization help?**

**Answer:**
1. **Prevents overfitting:** Keeps latent factors small
2. **Handles sparsity:** Prevents extreme values when few ratings
3. **Improves generalization:** Better predictions on unseen data

**Example:**
Without regularization: User with 1 rating → Latent factors can be arbitrarily large to fit that single point perfectly → Terrible generalization

**Q3: Calculate memory savings.**

**Given:** 1M users, 100K items, k=20

**Original matrix:** $10^6 \times 10^5 = 10^{11}$ entries  
**Factorized:** $10^6 \times 20 + 10^5 \times 20 = 22 \times 10^6$ entries

**Savings:** $\frac{10^{11}}{22 \times 10^6} \approx 4{,}545$× reduction!

### 4.3 Interview Questions

**Q (Netflix): Your matrix factorization model performs worse than item-item CF. Why might this happen?**

**A:** Several reasons:
1. **Too few factors:** $k$ too small, can't capture complexity
2. **Regularization too strong:** Over-smooths, loses signal
3. **Insufficient training:** Needs more epochs
4. **Data characteristics:** 
   - Very sparse data (< 0.1% ratings)
   - Strong item similarity (CF works well)
   - Cold-start heavy (many new users/items)
5. **Implementation issues:** Learning rate, initialization

**Solution:** Try hybrid approach - combine MF with CF

**Q (Amazon): How to handle implicit feedback (clicks, views) vs explicit ratings?**

**A:** Modify objective:

**Explicit:** Minimize $(r_{ij} - p_i \cdot q_j)^2$

**Implicit:** 
- Binary (clicked/not): Maximize $(p_i \cdot q_j) \times c_{ij}$ where $c_{ij} = 1$ if clicked
- Weighted: Use confidence $w_{ij} = 1 + \alpha \log(1 + \text{views})$

**ALS for Implicit:**
$$\min_{P,Q} \sum_{ij} w_{ij}(c_{ij} - p_i^T q_j)^2 + \lambda(\|P\|^2 + \|Q\|^2)$$

---

## Summary

**Key Takeaways:**
- Matrix Factorization decomposes $R \approx P \times Q^T$
- Learns $k$ latent factors automatically
- SGD optimizes by minimizing reconstruction error
- Regularization prevents overfitting
- Massive memory and computation savings
- Better handles sparsity than similarity-based CF

**Limitations:**
- Requires tuning ($k$, $\lambda$, learning rate)
- Cannot explain recommendations easily
- Cold-start still problematic
- Needs retraining for new data

**When to Use:**
- Large-scale systems (millions of users/items)
- Very sparse data
- Batch recommendation updates
- Accuracy is priority over interpretability

**Next Steps:** Explore SVD (Class 14) and ALS (Class 15) optimization methods
