# Class 02: Dimensionality Reduction using t-SNE

> **Core Principle:** "Preserving local neighborhoods in lower dimensions"

---

## Table of Contents
1. [Limitations of PCA](#1-limitations-of-pca)
2. [t-SNE Theory](#2-t-sne-theory)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [The Crowding Problem](#4-the-crowding-problem)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Limitations of PCA

### 1.1 The Linear Constraint

**Problem:** PCA only captures linear relationships through orthogonal projections.

**Analogy:** Imagine a Swiss roll (spiral structure). PCA is like crushing it flat with a rolling pin - the intrinsic structure is destroyed.

```
Original Swiss Roll           PCA Projection
    (3D spiral)              (2D crushed mess)
    
     ╔═══╗                        * * *
    ╔╝   ╚╗                      * * * *
   ╔╝     ╚╗      ---------->   * * * * *
  ╔╝       ╚╗                    * * * *
 ╔╝         ╚╗                    * * *
(Spiral intact)              (Structure lost)
```

### 1.2 When PCA Fails

**Example Scenarios:**
1. **Manifold Data**: Data lying on curved surfaces
2. **Clustered Data**: Multiple distinct groups with non-linear boundaries
3. **Non-Gaussian Distributions**: Highly skewed or multimodal data

**Mathematical Limitation:**  
PCA finds projection $\mathbf{y} = \mathbf{W}^T\mathbf{x}$ that maximizes variance. This assumes:
- Linear correlations matter most
- Gaussian-like distributions
- Global structure > local structure

---

## 2. t-SNE Theory

### 2.1 Core Philosophy

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** focuses on preserving **local neighborhoods** rather than global variance.

**Key Insight:** If point A and point B are neighbors in high dimensions, they should remain neighbors in low dimensions.

### 2.2 The Two-Step Process

**Step 1: Model High-Dimensional Similarities**  
For each pair of points in original space, calculate: "How likely would point $i$ pick point $j$ as its neighbor?"

**Step 2: Model Low-Dimensional Similarities**  
In the reduced space, measure: "How likely would point $i$ pick point $j$ as neighbor?"

**Objective:** Make these two probability distributions as similar as possible.

### 2.3 Visual Comparison

```
PCA Approach:                     t-SNE Approach:
"Maximize spread"                 "Preserve neighbors"

   Cluster A    Cluster B            Cluster A    Cluster B
      ●●●         ●●●                   ●●●         ●●●
      ●●●         ●●●                   ●●● ←same→  ●●●
      ●●●         ●●●                   ●●● groups  ●●●
       ↓                                 ↓
   (May mix clusters)              (Keeps clusters apart)
```

---

## 3. Mathematical Formulation

### 3.1 High-Dimensional Similarities

**Conditional Probability:**

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

**Interpretation:**
- Numerator: Gaussian similarity between $\mathbf{x}_i$ and $\mathbf{x}_j$
- Denominator: Normalization over all other points
- $\sigma_i$: Bandwidth parameter (controls neighborhood size)

**Symmetrized Joint Probability:**

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

### 3.2 Low-Dimensional Similarities

**Student's t-Distribution (1 degree of freedom):**

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**Why Student's t?** Heavy tails allow dissimilar points to be placed far apart without penalty.

### 3.3 Cost Function: Kullback-Leibler Divergence

**Objective:** Minimize the difference between $P$ and $Q$:

$$C = \text{KL}(P \| Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Gradient Descent Update:**

$$\frac{\delta C}{\delta \mathbf{y}_i} = 4 \sum_j (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

### 3.4 The Perplexity Parameter

**Definition:** A measure of the effective number of neighbors.

$$\text{Perplexity} = 2^{H(P_i)}$$

where $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$ is the Shannon entropy.

**Interpretation:**
- **Low perplexity (5-10)**: Focus on very local structure (tight clusters)
- **High perplexity (30-50)**: Consider broader neighborhoods

**Selection Rule:** Typical range is 5-50. For large datasets, use 30-50.

---

## 4. The Crowding Problem

### 4.1 Problem Statement

**Issue:** In high dimensions, there's more "room" to place points equidistantly. When reducing to 2D, points get "crowded."

**Example:**
- **10D space**: Can place 10 points equidistant from origin
- **2D space**: Can only place ~6 points equidistant from origin (hexagon)

### 4.2 Mathematical Explanation

**Volume of d-dim sphere:** $V_d \propto r^d$

As $d$ increases, most volume concentrates in a thin shell near the surface. When projecting to low dimensions, this shell must collapse, causing crowding.

### 4.3 t-SNE's Solution

**Student's t-Distribution (Heavy Tails):**

```
Gaussian:          Student's t:
   ●                    ●
  ●●●                  ● ●
 ●●●●●     vs.        ●   ●
  ●●●                  ● ●
   ●                    ●
(Forced together)   (Can spread apart)
```

The heavy tails of the t-distribution allow dissimilar points to be placed far

 apart in low dimensions without incurring large costs.

**Mathematical Proof:**

For large distances:
- Gaussian: $\exp(-d^2) \approx 0$ very quickly
- Student's t: $(1 + d^2)^{-1}$ decays much slower

---

## 5. Implementation

### 5.1 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Standardize (recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,          # Typical: 5-50
    learning_rate=200,      # Typical: 10-1000
    n_iter=1000,            # More iterations = better convergence
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE: 64D → 2D')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

### 5.2 Hyperparameter Tuning

```python
# Experiment with different perplexities
perplexities = [5, 10, 30, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, perp in enumerate(perplexities):
    ax = axes[idx // 3, idx % 3]
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)
    
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', s=10)
    ax.set_title(f'Perplexity = {perp}')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. Exam Preparation

### 6.1 Key Formulas Summary

| Concept | Formula | Notes |
|---------|---------|-------|
| High-D Similarity | $p_{j\|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$ | Gaussian kernel |
| Low-D Similarity | $q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$ | Student's t (df=1) |
| Cost Function | $\text{KL}(P \| Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$ | Minimize with gradient descent |
| Perplexity | $2^{H(P_i)}$ | Effective number of neighbors |

### 6.2 Common Exam Questions

**Q1: Why does t-SNE use Student's t-distribution in low dimensions?**  
**Answer:** To solve the crowding problem. The heavy tails of the t-distribution allow dissimilar points to be placed far apart without excessive cost, preventing clusters from collapsing into each other.

**Q2: What is the computational complexity of t-SNE?**  
**Answer:** $O(n^2)$ in naive implementation due to pairwise distance calculations. With optimizations (Barnes-Hut approximation), it reduces to $O(n \log n)$.

**Q3: Can t-SNE be used for new data points?**  
**Answer:** No, standard t-SNE does not have a `transform()` method. Each run requires refitting the entire dataset. For out-of-sample extension, use parametric t-SNE or UMAP instead.

**Q4: Compare PCA and t-SNE.**

| Aspect | PCA | t-SNE |
|--------|-----|-------|
| Type | Linear | Non-linear |
| Objective | Maximize variance | Preserve local neighborhoods |
| Speed | Fast ($O(d^2n + d^3)$) | Slow ($O(n^2)$ or $O(n \log n)$) |
| Deterministic | Yes | No (random initialization) |
| Global structure | Preserves | May distort |
| New data | Can transform | Must refit |
| Interpretability | High | Low |

### 6.3 Numerical Problem

**Problem:** Given high-dimensional similarity $p_{12} = 0.35$ and low-dimensional similarity $q_{12} = 0.15$, compute the contribution to KL divergence.

**Solution:**

$$C_{12} = p_{12} \log \frac{p_{12}}{q_{12}} = 0.35 \times \log \frac{0.35}{0.15}$$

$$= 0.35 \times \log(2.33) = 0.35 \times 0.846 = 0.296$$

**Interpretation:** This positive value indicates gradient will push points 1 and 2 closer in low-dimensional space.

### 6.4 Interview Questions

**Technical (Research Roles):**

**Q:** How does Barnes-Hut approximation speed up t-SNE?  
**A:** Instead of computing exact pairwise forces ($O(n^2)$), it uses quad-trees (2D) or oct-trees (3D) to approximate distant forces. Groups far-away points and treats them as a single center of mass, reducing complexity to $O(n \log n)$.

**Q:** Why is t-SNE non-deterministic?  
**A:** Due to random initialization of low-dimensional points and stochastic gradient descent. Different runs can produce different layouts (though cluster structures should remain consistent).

**Practical (Industry):**

**Q:** When would you use t-SNE vs. PCA?  
**A:**
- **Use PCA:** Feature extraction for ML pipelines, real-time applications, preserving global structure, interpretability needed
- **Use t-SNE:** Exploratory visualization, understanding cluster structures, presentation graphics, when local relationships matter more than global

**Q:** A user reports t-SNE results look different each time. Is this a bug?  
**A:** No, this is expected behavior. Set `random_state` for reproducibility. If structure changes drastically, increase iterations or adjust perplexity.

---

## Summary

**Key Takeaways:**
1. t-SNE preserves local neighborhoods using probabilistic similarities
2. Uses Gaussian kernel in high-D, Student's t in low-D (solves crowding)
3. Optimizes via gradient descent on KL divergence
4. Perplexity controls neighborhood size (typically 30-50)
5. Non-deterministic, no out-of-sample extension

**Best Practices:**
- Always standardize input data
- Try multiple perplexity values
- Run for sufficient iterations (1000-5000)
- Set random_state for reproducibility
- Don't interpret distances between clusters as meaningful

**Limitations:**
- Computationally expensive ($O(n^2)$)
- No inverse transform
- May create false patterns if parameters poorly chosen
- Global structure not preserved
- Different runs yield different results
