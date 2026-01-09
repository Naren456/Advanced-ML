# Class 04: Clustering - K-Means and K-Means++

> **Core Principle:** "Partitioning data by minimizing within-cluster variance"

---

## Table of Contents
1. [Clustering Fundamentals](#1-clustering-fundamentals)
2. [K-Means Algorithm](#2-k-means-algorithm)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [K-Means++ Initialization](#4-k-means-initialization)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Implementation](#6-implementation)
7. [Exam Preparation](#7-exam-preparation)

---

## 1. Clustering Fundamentals

### 1.1 Unsupervised Learning

**Definition:** Finding hidden patterns in unlabeled data without predefined categories.

**Analogy:** Organizing a library without a cataloging system. Books naturally group by topic, author style, and language - but you must discover these groups yourself.

### 1.2 Clustering Objectives

**Intra-cluster Similarity:** Points within the same cluster should be similar  
**Inter-cluster Dissimilarity:** Points in different clusters should be different

**Mathematical Expression:**

$$\text{Maximize: } \frac{\text{Inter-cluster distance}}{\text{Intra-cluster distance}}$$

### 1.3 Types of Clustering

```
Hard Clustering (K-Means):      Soft Clustering (GMM):
      Cluster A  Cluster B            Cluster A  Cluster B
         ●●●       ●●●                   ●●●●      ●●●●
         ●●●       ●●●                   ●●●●      ●●●●
         ●●●       ●●●                   ●●●●      ●●●●
    (Definite boundaries)           (Probabilistic membership)
```

---

## 2. K-Means Algorithm

### 2.1 Intuitive Explanation

**Analogy:** Imagine opening $K$ pizza delivery hubs in a city:

**Goal:** Position hubs so average delivery distance is minimized

**Process:**
1. **Initialize:** Place $K$ hubs randomly
2. **Assign:** Each customer calls nearest hub (creates service zones)
3. **Update:** Move each hub to exact center of its zone
4. **Iterate:** Customers may now be closer to different hubs - reassign and repeat

### 2.2 Algorithm Steps

**Step 1: Initialization**  
Randomly select $K$ points from dataset as initial centroids: $C = \{c_1, c_2, \ldots, c_K\}$

**Step 2: Assignment (E-Step)**  
For each point $x_i$, assign to nearest centroid:

$$\text{cluster}(x_i) = \argmin_{j \in \{1,\ldots,K\}} \|x_i - c_j\|^2$$

**Step 3: Update (M-Step)**  
Recompute centroids as mean of assigned points:

$$c_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$

where $S_j$ is the set of points assigned to cluster $j$.

**Step 4: Convergence**  
Repeat Steps 2-3 until:
- Centroids don't change: $\|c_j^{(t+1)} - c_j^{(t)}\| < \epsilon$
- Maximum iterations reached
- No points change clusters

### 2.3 Visual Example

**Iteration 0 (Initialization):**
```
    y
    |
  4 |     *              * = data point
    |  *     *           + = centroid
  3 | *   +1      *
    |*       *  +2
  2 |   *    *
    | *   +3
  1 |     *
    |________________x
    0  1  2  3  4  5
```

**Iteration 1 (After first assignment & update):**
```
    y
    |
  4 |     *
    |  ●     ●          ● = cluster 1
  3 | ●   +1      ◆    ◆ = cluster 2
    |●       ◆  +2     ■ = cluster 3
  2 |   ■    ◆
    | ■   +3
  1 |     ■
    |________________x
```

**Iteration 3 (Converged):**
```
Centroid positions stabilized
No points change clusters
```

---

## 3. Mathematical Formulation

### 3.1 Objective Function

**Minimize Within-Cluster Sum of Squares (WCSS or Inertia):**

$$J(C, S) = \sum_{j=1}^{K} \sum_{x_i \in S_j} \|x_i - c_j\|^2$$

where:
- $C = \{c_1, \ldots, c_K\}$: centroids
- $S = \{S_1, \ldots, S_K\}$: cluster assignments

**Equivalently (minimizing variance):**

$$J = \sum_{j=1}^{K} |S_j| \cdot \text{Var}(S_j)$$

### 3.2 Convergence Proof

**Theorem:** K-Means converges to a local minimum.

**Proof Sketch:**

1. **Assignment step decreases $J$:** Assigning each point to nearest centroid minimizes distance

2. **Update step decreases $J$:** For cluster $S_j$, the point that minimizes sum of squared distances is the mean:

   $$\frac{d}{dc_j}\sum_{x_i \in S_j}\|x_i - c_j\|^2 = 0$$
   
   $$\implies c_j = \frac{1}{|S_j|}\sum_{x_i \in S_j} x_i$$

3. **$J$ is bounded below** by 0

4. **Monotonic decrease + bounded** → convergence guaranteed

**Note:** Convergence is to local minimum, not necessarily global minimum.

### 3.3 Computational Complexity

**Time Complexity:** $O(n \cdot K \cdot d \cdot I)$

where:
- $n$: number of data points
- $K$: number of clusters
- $d$: dimensionality
- $I$: number of iterations (typically log converges)

**Space Complexity:** $O((n+K) \cdot d)$

---

## 4. K-Means++ Initialization

### 4.1 The Initialization Problem

**Standard K-Means Issue:** Random initialization can lead to poor local minima.

**Example:**
```
True clusters:          Bad initialization:      Poor result:
  ●●●     ◆◆◆              ●●●+1  +2◆◆◆            ●●●●●◆◆◆
  ●●●     ◆◆◆     →        ●●●     +3◆◆◆    →      ●●●●●◆◆◆
  ●●●     ◆◆◆              ●●●       ◆◆◆            ●●●●●◆◆◆
(Well separated)      (All centroids right)     (One big cluster)
```

### 4.2 K-Means++ Algorithm

**Principle:** Choose initial centroids that are far apart from each other.

**Algorithm:**

**Step 1:** Choose first centroid $c_1$ uniformly at random from data points

**Step 2:** For each subsequent centroid $c_i$ ($i = 2, \ldots, K$):
   
   a) Compute distance from each point $x$ to nearest existing centroid:
   
   $$D(x) = \min_{j < i} \|x - c_j\|^2$$
   
   b) Choose next centroid with probability proportional to $D(x)^2$:
   
   $$P(x \text{ is chosen}) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$$

**Step 3:** Proceed with standard K-Means

**Intuition:** Points far from existing centroids are more likely to be selected, ensuring good coverage.

### 4.3 Theoretical Guarantee

**Theorem (Arthur & Vassilvitskii, 2007):**

K-Means++ initialization guarantees:

$$\mathbb{E}[J] \leq 8(\ln K + 2) \cdot J_{opt}$$

where $J_{opt}$ is the globally optimal clustering.

**Practical Impact:**
- Faster convergence (fewer iterations)
- Better final clustering quality
- More consistent results across runs

---

## 5. Evaluation Metrics

### 5.1 Within-Cluster Sum of Squares (WCSS)

**Formula:**

$$\text{WCSS} = \sum_{j=1}^{K} \sum_{x_i \in S_j} \|x_i - c_j\|^2$$

**Problem:** Always decreases as $K$ increases

$$K = n \implies \text{WCSS} = 0$$

### 5.2 Elbow Method

**Procedure:**
1. Run K-Means for $K \in \{1, 2, \ldots, K_{\max}\}$
2. Plot WCSS vs. $K$
3. Look for "elbow" - point where WCSS reduction slows

**Visual:**
```
WCSS
  |
100|●
  |
 50|  ●
  |    ●
 20|      ●──── Elbow (K=4)
  |         ●●●●●●
  0|________________K
    1  2  3  4  5  6  7
```

**Interpretation:** Beyond the elbow, adding clusters provides diminishing returns.

### 5.3 Silhouette Score

**For each point $i$:**

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

where:
- $a(i)$: mean distance to other points in same cluster (cohesion)
- $b(i)$: mean distance to points in nearest different cluster (separation)

**Silhouette value interpretation:**
- $s(i) \approx 1$: Well clustered
- $s(i) \approx 0$: On cluster boundary
- $s(i) < 0$: Possibly in wrong cluster

**Average Silhouette Score:**

$$S = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Optimal $K$:** Maximizes average silhouette score

### 5.4 Davies-Bouldin Index

**Formula:**

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left(\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}\right)$$

where:
- $\sigma_i$: average distance of points in cluster $i$ to centroid $c_i$
- $d(c_i, c_j)$: distance between centroids

**Interpretation:** Lower is better (tight clusters that are far apart)

---

## 6. Implementation

### 6.1 Basic K-Means

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, 
                       cluster_std=0.60, random_state=0)

# Standardize (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', s=200, marker='X', edgecolors='black', linewidth=2)
plt.title('K-Means Clustering (K=4)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

### 6.2 Elbow Method Implementation

```python
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method For Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# Alternatively, use KneeLocator
from kneed import KneeLocator
kn = KneeLocator(list(K_range), wcss, curve='convex', direction='decreasing')
print(f"Optimal K: {kn.knee}")
```

### 6.3 Silhouette Analysis

```python
from sklearn.metrics import silhouette_score, silhouette_samples

K_range = range(2, 11)
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True, alpha=0.3)
plt.show()
```

### 6.4 Detailed Silhouette Visualization

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
K_values = [2, 3, 4, 5]

for idx, k in enumerate(K_values):
    ax = axes[idx // 2, idx % 2]
    
    # Fit
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Silhouette values
    silhouette_vals = silhouette_samples(X_scaled, labels)
    avg_score = silhouette_score(X_scaled, labels)
    
    # Plot
    y_lower = 10
    for i in range(k):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size
        
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_vals,
                         facecolor=color, alpha=0.7)
        
        y_lower = y_upper + 10
    
    ax.axvline(x=avg_score, color="red", linestyle="--", label=f'Avg: {avg_score:.2f}')
    ax.set_title(f'K = {k}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## 7. Exam Preparation

### 7.1 Key Formulas Summary

| Concept | Formula | Notes |
|---------|---------|-------|
| Objective | $\min \sum_{j=1}^{K} \sum_{x \in S_j} \|x - c_j\|^2$ | Minimize intra-cluster variance |
| Assignment | $\argmin_j \|x_i - c_j\|^2$ | Assign to nearest centroid |
| Update | $c_j = \frac{1}{|S_j|}\sum_{x \in S_j} x$ | Centroid = mean of cluster |
| Silhouette | $s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$ | Range: [-1, 1], higher better |

### 7.2 Step-by-Step Example

**Given:** Points $\{(1,2), (2,3), (8,8), (9,10)\}$, $K=2$

**Initial centroids:** $c_1 = (1,2)$, $c_2 = (8,8)$

**Iteration 1:**

*Assignment:*
- Point $(1,2)$: dist to $c_1 = 0$, dist to $c_2 = \sqrt{85}$ → Cluster 1
- Point $(2,3)$: dist to $c_1 = \sqrt{2}$, dist to $c_2 = \sqrt{61}$ → Cluster 1  
- Point $(8,8)$: dist to $c_1 = \sqrt{85}$, dist to $c_2 = 0$ → Cluster 2
- Point $(9,10)$: dist to $c_1 = \sqrt{128}$, dist to $c_2 = \sqrt{5}$ → Cluster 2

*Update:*
$$c_1 = \frac{(1,2) + (2,3)}{2} = (1.5, 2.5)$$
$$c_2 = \frac{(8,8) + (9,10)}{2} = (8.5, 9.0)$$

**Iteration 2:**

Repeat assignment with new centroids... (converges quickly in this case)

### 7.3 Common Exam Questions

**Q1: Why is K-Means sensitive to initialization?**

**Answer:** K-Means converges to local minima. Different initializations lead to different local minima. Poor initialization (e.g., all centroids in one true cluster) can result in suboptimal clustering.

Example: If all initial centroids are chosen from one true cluster, K-Means may never discover other clusters.

**Q2: What are the assumptions and limitations of K-Means?**

**Assumptions:**
1. Clusters are spherical (isotropic Gaussian)
2. Clusters have similar sizes
3. Clusters have similar densities
4. Euclidean distance is meaningful metric

**Limitations:**
1. Must specify $K$ beforehand
2. Sensitive to outliers (mean is affected)
3. Assumes spherical clusters (fails on elongated or irregular shapes)
4. Performance degrades in high dimensions (curse of dimensionality)

**Q3: Compare K-Means and K-Means++.**

| Aspect | K-Means | K-Means++ |
|--------|---------|-----------|
| Initialization | Random | Smart (spread out) |
| Convergence speed | Variable | Faster |
| Final quality | Inconsistent | More consistent |
| Theoretical guarantee | None | $O(\log K)$ approximation |
| Complexity | $O(nKd)$ | $O(nKd + K^2d)$ |

**Q4: How to choose $K$?**

**Methods:**
1. **Elbow Method:** Plot WCSS vs. $K$, find elbow
2. **Silhouette Score:** Maximize average silhouette
3. **Gap Statistic:** Compare WCSS to random data
4. **Domain Knowledge:** Business requirements or data understanding

### 7.4 Interview Questions

**Technical (FAANG):**

**Q:** You have customer purchase data with different scales (age: 20-80, income: 20k-200k). What preprocessing is needed before K-Means?

**A:** **Standardization is mandatory.** Without it, income will dominate distance calculations purely due to scale. Use StandardScaler:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This ensures each feature contributes equally to distance computations.

**Q:** K-Means fails on crescent-shaped clusters. What alternatives exist?

**A:**
1. **DBSCAN:** Density-based, handles arbitrary shapes
2. **Spectral Clustering:** Uses graph theory, good for non-convex clusters
3. **GMM:** Gaussian Mixture Models for elliptical clusters
4. **Kernel K-Means:** Projects data to higher dimension where clusters become spherical

**Practical (Retail/E-commerce):**

**Q:** We clustered customers but cluster sizes are very unbalanced (1% in one cluster, 50% in another). Is this a problem?

**A:** Depends on application:
- **Problem if:** You need balanced segments for A/B testing or resource allocation
- **Not a problem if:** Natural data distribution (e.g., VIP customers are rare)
- **Solutions:** 
  - Try different $K$ values
  - Use hierarchical clustering
  - Consider business constraints in algorithm choice

---

## Summary

**Key Takeaways:**
1. K-Means partitions data by minimizing within-cluster variance
2. Algorithm alternates between assignment and update steps
3. K-Means++ initialization significantly improves results
4. Multiple methods exist for choosing optimal $K$
5. Always standardize data before clustering

**Best Practices:**
- Use K-Means++ initialization
- Try multiple values of $K$
- Standardize features
- Run multiple times with different seeds
- Validate with silhouette score
- Consider domain knowledge

**When to Use K-Means:**
- Clear number of clusters expected
- Clusters are roughly spherical
- Large datasets (fast algorithm)
- Interpretability important (centroids as representatives)

**When NOT to Use K-Means:**
- Irregular cluster shapes
- Widely varying cluster sizes/densities
- Heavy outliers present
- Unknown optimal $K$
