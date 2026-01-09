# Class 03: Advanced Dimensionality Reduction - t-SNE Deep Dive and UMAP

> **Core Principle:** "Balancing local and global structure in manifold learning"

---

## Table of Contents
1. [t-SNE: Advanced Concepts](#1-t-sne-advanced-concepts)
2. [UMAP Theory](#2-umap-theory)
3. [Comparative Analysis](#3-comparative-analysis)
4. [Implementation](#4-implementation)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. t-SNE: Advanced Concepts

### 1.1 Mathematical Deep Dive

**The Optimization Landscape:**

t-SNE minimizes KL divergence through gradient descent. The gradient with respect to low-dimensional point $\mathbf{y}_i$ is:

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4\sum_{j} (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)Z_{ij}$$

where $Z_{ij} = (1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$

**Physical Interpretation:**
- $(p_{ij} - q_{ij})$: Attractive/repulsive force strength
- $(\mathbf{y}_i - \mathbf{y}_j)$: Direction of force
- $Z_{ij}$: Distance-dependent scaling


### 1.2 Perplexity: The Critical Hyperparameter

**Mathematical Definition:**

For each point $i$, we solve for $\sigma_i$ such that:

$$\text{Perp}(P_i) = 2^{H(P_i)} = 2^{-\sum_j p_{j|i} \log_2 p_{j|i}}$$

**Effect on Results:**

```
Low Perplexity (5):        Medium Perplexity (30):      High Perplexity (100):
  ●●  ●●  ●●                    ●●●● ●●●●                  ●●●●●●●●●●
  ●●  ●●  ●●                    ●●●● ●●●●                  ●●●●●●●●●●
  ●●  ●●  ●●                    ●●●● ●●●●                  ●●●●●●●●●●
(Many tiny clusters)         (Balanced structure)       (May merge clusters)
```

**Selection Guidelines:**
- $n < 100$: perplexity = 5-10
- $100 < n < 1000$: perplexity = 30
- $n > 1000$: perplexity = 30-50

### 1.3 Learning Rate Dynamics

**Formula:**

$$\mathbf{y}^{(t+1)}_i = \mathbf{y}^{(t)}_i + \eta \frac{\partial C}{\partial \mathbf{y}_i} + \alpha(t)(\mathbf{y}^{(t)}_i - \mathbf{y}^{(t-1)}_i)$$

where:
- $\eta$: learning rate (typically 100-1000)
- $\alpha(t)$: momentum term

**Adaptive Learning:**
```python
if iteration < 250:
    momentum = 0.5    # Early exploration
else:
    momentum = 0.8    # Later refinement
```

---

## 2. UMAP Theory

### 2.1 Conceptual Foundation

**UMAP (Uniform Manifold Approximation and Projection)** is based on Riemannian geometry and algebraic topology.

**Core Assumption:** Data is uniformly distributed on a locally connected Riemannian manifold.

**Analogy:** Imagine Earth's surface. Locally (your neighborhood), it appears flat. Globally, it's spherical. UMAP preserves both local flatness and global curvature.

### 2.2 Mathematical Framework

**Step 1: Construct Fuzzy Topological Representation**

High-dimensional membership strength:

$$w(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\max(0, d_{ij} - \rho_i)}{\sigma_i}\right)$$

where:
- $d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$
- $\rho_i$: distance to nearest neighbor
- $\sigma_i$: normalizing factor

**Step 2: Symmetrize with Fuzzy Union**

$$A = A_{high} + A_{high}^T - A_{high} \circ A_{high}^T$$

**Step 3: Optimize Low-Dimensional Layout**

Minimize cross-entropy:

$$C = \sum_{ij} \left[a_{ij} \log\frac{a_{ij}}{b_{ij}} + (1-a_{ij})\log\frac{1-a_{ij}}{1-b_{ij}}\right]$$

where:

$$b_{ij} = \frac{1}{1 + \alpha d_{ij}^{2\beta}}$$

Default: $\alpha = 1$, $\beta = 1$

### 2.3 Key Parameters

**1. n_neighbors (default: 15)**

Analogous to perplexity in t-SNE.

$$\text{n\_neighbors} \approx 3 \times \text{perplexity}$$

**Effect:**
- **Low (5-10)**: Emphasizes local structure, many small clusters
- **High (50-100)**: Emphasizes global structure, broader patterns

**2. min_dist (default: 0.1)**

Controls how tightly points pack together.

$$\text{min\_dist} \in [0.0, 0.99]$$

**Effect:**
- **0.0**: Very tight clusters (good for cluster detection)
- **0.5-0.99**: Looser spreads (better for visualization)

---

## 3. Comparative Analysis

### 3.1 PCA vs t-SNE vs UMAP

| Feature | PCA | t-SNE | UMAP |
|---------|-----|-------|------|
| **Method** | Linear projection | Probabilistic | Topological |
| **Complexity** | $O(d^2n + d^3)$ | $O(n^2)$ or $O(n\log n)$ | $O(n^{1.14})$ |
| **Speed** | Very fast | Slow | Fast |
| **Deterministic** | Yes | No | Reproducible with seed |
| **Local Structure** | No | Excellent | Excellent |
| **Global Structure** | Yes | Poor | Good |
| **Scalability** | Excellent | Poor | Excellent |
| **Parameters** | n_components | perplexity, lr | n_neighbors, min_dist |
| **Out-of-sample** | Yes | No | Yes (with model) |
| **Interpretability** | High | Low | Low |

### 3.2 Visual Comparison Example

**Dataset:** 3D Swiss Roll embedded in 64D space

```
Ground Truth:            PCA:                 t-SNE:               UMAP:
    ╔═══╗                ▓▓▓▓▓                 ●●● ●●●             ●●● ●●●
   ╔╝   ╚╗              ▓▓▓▓▓▓▓               ●●   ●●            ●●   ●●
  ╔╝     ╚╗        →    ▓▓▓▓▓▓▓          →   ●     ●       →    ●     ●
 ╔╝       ╚╗            ▓▓▓▓▓▓▓               ●●   ●●            ●●   ●●
╔╝         ╚╗            ▓▓▓▓▓                 ●●● ●●●             ●●● ●●●
(Spiral)              (Crushed)          (Local preserved)    (Balanced)
```

### 3.3 When to Use Each

**Use PCA when:**
- Speed is critical
- Interpretability needed
- Feature engineering for ML pipeline
- Data is approximately linear

**Use t-SNE when:**
- Visualization is primary goal
- Local clusters are most important
- Small to medium datasets (< 10,000 points)
- Don't need to transform new data

**Use UMAP when:**
- Need both local and global structure
- Large datasets (> 10,000 points)
- Require out-of-sample projection
- Want reproducible results
- Speed matters

---

## 4. Implementation

### 4.1 UMAP Implementation

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Basic UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,       # Balance local/global
    min_dist=0.1,         # Cluster tightness
    metric='euclidean',
    random_state=42
)

embedding = reducer.fit_transform(X_scaled)

# Visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                     c=y, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('UMAP Projection of Digits Dataset')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
```

### 4.2 Hyperparameter Grid Search

```python
from sklearn.metrics import silhouette_score

# Parameter grid
n_neighbors_list = [5, 15, 30, 50]
min_dist_list = [0.0, 0.1, 0.3, 0.5]

results = []

for n_neighbors in n_neighbors_list:
    for min_dist in min_dist_list:
        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embedding = reducer.fit_transform(X_scaled)
       
        # Evaluate (using known labels for demonstration)
        score = silhouette_score(embedding, y)
        results.append({
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'silhouette': score
        })

# Best configuration
best = max(results, key=lambda x: x['silhouette'])
print(f"Best: n_neighbors={best['n_neighbors']}, "
      f"min_dist={best['min_dist']}, score={best['silhouette']:.3f}")
```

### 4.3 Comparing All Three Methods

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Spectral', s=5)
axes[0].set_title(f'PCA\nVariance: {sum(pca.explained_variance_ratio_):.2%}')

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Spectral', s=5)
axes[1].set_title('t-SNE\nPerplexity: 30')

# UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=5)
axes[2].set_title('UMAP\nn_neighbors: 15')

plt.tight_layout()
plt.show()
```

---

## 5. Exam Preparation

### 5.1 Key Formulas Summary

| Method | Core Formula | Parameters |
|--------|--------------|------------|
| **t-SNE** | $\min \text{KL}(P \| Q)$ | perplexity, learning_rate |
| **UMAP** | $\min \text{CE}(A, B)$ | n_neighbors, min_dist |

### 5.2 Common Exam Questions

**Q1: How does UMAP achieve better scalability than t-SNE?**

**Answer:** 
- **t-SNE:** Computes full $n \times n$ similarity matrix → $O(n^2)$
- **UMAP:** Builds approximate k-nearest neighbor graph → $O(n \log n)$ with efficient algorithms (NN-Descent)
- UMAP uses stochastic gradient descent with negative sampling, processing only a subset of edges per iteration

**Q2: Explain the relationship between n_neighbors in UMAP and perplexity in t-SNE.**

**Answer:**
Both control the scale of local neighborhoods:
- **Perplexity:** Effective number of neighbors in probabilistic sense
- **n_neighbors:** Actual number of neighbors in graph construction
- Rule of thumb: $\text{n\_neighbors} \approx 3 \times \text{perplexity}$

**Q3: Why does UMAP preserve global structure better than t-SNE?**

**Answer:**
1. **Fuzzy topology:** UMAP models data as fuzzy simplicial complex, capturing global manifold structure
2. **Cross-entropy loss:** Unlike KL divergence, it's symmetric and penalizes both false attractions and false repulsions equally
3. **Graph construction:** Uses approximate nearest neighbors to maintain long-range connections

### 5.3 Numerical Problem

**Problem:** Given UMAP parameters $\alpha = 1$, $\beta = 1$, compute low-dimensional similarity for points at distance $d = 2$.

**Solution:**

$$b = \frac{1}{1 + \alpha d^{2\beta}} = \frac{1}{1 + 1 \cdot 2^{2 \cdot 1}} = \frac{1}{1 + 4} = 0.2$$

**Interpretation:** Points at distance 2 have membership strength 0.2 in the low-dimensional fuzzy set.

### 5.4 Interview Questions

**Research/Academic:**

**Q:** Describe the theoretical foundations of UMAP.

**A:** UMAP is grounded in:
1. **Riemannian geometry:** Assumes data lies on a locally varying metric space
2. **Algebraic topology:** Uses simplicial sets to represent data structure
3. **Category theory:** Fuzzy singular set functor provides the mathematical framework

The key insight: There exists a low-dimensional representation with equivalent fuzzy topological structure to the high-dimensional data.

**Industry/Applied:**

**Q:** You're visualizing customer segments. Should you use t-SNE or UMAP?

**A:** Use **UMAP** because:
1. Customer data is typically large (> 10k records) → UMAP is faster
2. Business needs reproducible results → UMAP is deterministic with fixed seed
3. May need to project new customers → UMAP supports transform()
4. Want both cluster separation (local) and relationships between segments (global) → UMAP balances both

**Q:** How do you validate dimensionality reduction results?

**A:**
1. **Quantitative:**
   - Silhouette score (if labels available)
   - Trustworthiness and continuity metrics
   - Shepard diagram (original vs. embedded distances)

2. **Qualitative:**
   - Visual inspection of cluster quality
   - Domain expert review
   - Stability across multiple runs
   - Sensitivity analysis on hyperparameters

---

## Summary

**t-SNE Strengths:**
- Excellent local structure preservation
- Well-studied and widely adopted
- Clear probabilistic interpretation

**t-SNE Limitations:**
- Slow on large datasets
- Non-deterministic
- Poor global structure
- No out-of-sample extension

**UMAP Strengths:**
- Fast and scalable
- Preserves both local and global structure
- Reproducible with seed
- Supports new data transformation
- Flexible distance metrics

**UMAP Limitations:**
- More complex theoretical foundation
- Newer algorithm (less established)
- Hyperparameter tuning can be tricky

**Practical Recommendations:**
- **Exploration:** Start with UMAP for quick overview
- **Publication:** Use multiple methods and show consistency
- **Production:** UMAP for scalability and reproducibility
- **Understanding:** t-SNE for detailed cluster analysis
