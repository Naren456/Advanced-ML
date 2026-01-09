# Class 06: Density-Based Clustering - DBSCAN

> **Core Principle:** "Clustering based on density, not distance"

---

## Table of Contents
1. [Limitations of Centroid-Based Methods](#1-limitations-of-centroid-based-methods)
2. [DBSCAN Theory](#2-dbscan-theory)
3. [Algorithm](#3-algorithm)
4. [Mathematical Analysis](#4-mathematical-analysis)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Limitations of Centroid-Based Methods

### 1.1 The Spherical Assumption Problem

**K-Means and GMM** assume clusters are convex (roughly spherical/elliptical)

**Visual Examples Where They Fail:**

```
Concentric Circles:           Crescent Moons:           S-Curve:
    ○○○○○                       ●●●                      ●●
   ○    ○                     ●●  ●●                   ●●
  ○  ●●  ○                   ●      ●                 ●●
   ○ ●● ○                   ● Source  ●●             ●●
    ○○○○○                     ●●●●●  ●              ●●
(Two clusters)              (Two clusters)         (One cluster)
```

**K-Means Result:** Incorrect split through the middle

### 1.2 Outlier Sensitivity

**Problem:** K-Means forces ALL points into clusters

```
True Structure:             K-Means Result:
  ●●●     ●●●                 ●●●     ●●●
  ●●●  *  ●●●                 ●●●  *← ●●●
  ●●●     ●●●                 ●●●     ●●●
(* = noise)              (Noise pulls centroid)
```

---

## 2. DBSCAN Theory

### 2.1 Core Philosophy

**DBSCAN** = Density-Based Spatial Clustering of Applications with Noise

**Key Insight:** Clusters are dense regions separated by sparse regions

**Analogy:** Imagine a crowded city from aerial view:
- **Dense areas** (downtown): Many people close together → Cluster
- **Sparse areas** (highways): Few scattered people → Noise
- **Border areas** (suburbs): People near downtown edge → Border points

### 2.2 Fundamental Concepts

**Definition 1: Epsilon Neighborhood**  
The ε-neighborhood of point $p$:

$$N_\epsilon(p) = \{q \in D : \text{dist}(p, q) \leq \epsilon\}$$

**Definition 2: Core Point**  
A point $p$ is a core point if:

$$|N_\epsilon(p)| \geq \text{MinPts}$$

**Definition 3: Border Point**  
A point that is not a core point but lies in the ε-neighborhood of a core point

**Definition 4: Noise Point**  
A point that is neither core nor border

### 2.3 Density Reachability

**Directly Density-Reachable:**  
Point $q$ is directly density-reachable from $p$ if:
1. $q \in N_\epsilon(p)$
2. $p$ is a core point

**Density-Reachable:**  
$q$ is density-reachable from $p$ if there exists a chain:  
$p = p_1, p_2, \ldots, p_n = q$

where each $p_{i+1}$ is directly density-reachable from $p_i$

**Density-Connected:**  
Points $p$ and $q$ are density-connected if there exists a point $o$ such that both $p$ and $q$ are density-reachable from $o$

### 2.4 Visual Definitions

```
    ε = 2 units, MinPts = 4

Legend:
  ● = Core point (≥4 neighbors within ε)
  ○ = Border point (in ε-neighborhood of core, but <4 neighbors)
  × = Noise point (neither core nor border)

    2ε
    ├─┤
    ●──●──●──●       ○       ×
    │  │  │  │      /
    ●──●──●──●     ●
    │  │  │  │
    ●──●──●──●
    
Cluster A          Border   Noise
```

---

## 3. Algorithm

### 3.1 DBSCAN Algorithm

**Input:**
- Dataset $D$
- Parameters: $\epsilon$ (radius), MinPts (minimum points)

**Output:**
- Cluster labels for each point

**Procedure:**

```
Initialize all points as UNVISITED

FOR each point P in dataset:
    IF P is VISITED:
        CONTINUE
    
    MARK P as VISITED
    
    neighbors = GET_NEIGHBORS(P, ε)
    
    IF |neighbors| < MinPts:
        MARK P as NOISE
    ELSE:
        cluster_id = NEXT_CLUSTER_ID()
        EXPAND_CLUSTER(P, neighbors, cluster_id, ε, MinPts)

FUNCTION EXPAND_CLUSTER(P, neighbors, cluster_id, ε, MinPts):
    ADD P to cluster_id
    
    FOR each point Q in neighbors:
        IF Q is UNVISITED:
            MARK Q as VISITED
            Q_neighbors = GET_NEIGHBORS(Q, ε)
            
            IF |Q_neighbors| >= MinPts:
                neighbors = UNION(neighbors, Q_neighbors)
        
        IF Q does not belong to any cluster:
            ADD Q to cluster_id
```

### 3.2 Step-by-Step Example

**Dataset:** 10 points in 2D, $\epsilon = 1.5$, MinPts = 3

```
Step 1: Start with point A
        A has 3 neighbors (B, C, D) → Core point
        Create Cluster 1
        
Step 2: Expand from A
        Check B: has 2 neighbors → Border point, add to Cluster 1
        Check C: has 4 neighbors → Core point, add to Cluster 1
        
Step 3: Continue expansion from C
        Check E, F, G...
        
Step 4: Move to unvisited point X
        X has 1 neighbor → Noise
        
Final: Cluster 1 = {A, B, C, D, E, F, G}
       Noise = {X, Y}
```

---

## 4. Mathematical Analysis

### 4.1 Complexity Analysis

**Time Complexity:**

**Without index structure:**
- Finding neighbors for each point: $O(n^2)$
- Overall: $O(n^2)$

**With spatial index (e.g., R*-tree, kd-tree):**
- Neighbor queries: $O(\log n)$ each
- Overall: $O(n \log n)$

**Space Complexity:** $O(n)$

### 4.2 Parameter Selection

**ε (Epsilon) Selection:**

**K-distance Graph Method:**
1. For each point, compute distance to $k$-th nearest neighbor  
   (typically $k$ = MinPts)
2. Plot sorted k-distances
3. Look for "elbow" - sharp increase indicates noise threshold

```
k-dist
  |
  |              ╱
  |            ╱
  |          ╱← Elbow: choose this as ε
  |        ╱
  |╭╴╴╴╴╭╴
  |________ Points (sorted by k-dist)
```

**MinPts Selection:**

**Rule of Thumb:**
- MinPts $\geq d + 1$ where $d$ is dimensionality
- For 2D data: MinPts = 4 or 5
- For large datasets: MinPts = 2 × dimensionality

**Rationale:** Higher $d$ → need more points to define "density"

### 4.3 Properties and Theorems

**Theorem 1:** Let $p$ be a core point. Then the set of all points density-reachable from $p$ forms a cluster.

**Theorem 2:** DBSCAN finds all density-based clusters (given correct parameters).

**Theorem 3:** DBSCAN is deterministic except for border point assignment when equidistant from multiple clusters.

### 4.4 Advantages and Limitations

**Advantages:**
1. **Arbitrary Shapes:** Can find clusters of any shape
2. **No $K$ Required:** Automatically determines number of clusters
3. **Robust to Outliers:** Explicitly detects noise
4. **Single Scan:** (Mostly) one pass through data

**Limitations:**
1. **Varying Density:** Struggles when clusters have different densities
2. **Parameter Sensitivity:** Results highly dependent on $\epsilon$ and MinPts
3. **Curse of Dimensionality:** Distance becomes less meaningful in high dimensions
4. **Border Point Ambiguity:** Border points near multiple clusters assigned arbitrarily

---

## 5. Implementation

### 5.1 Basic DBSCAN

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate non-convex data
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# Standard scale (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Visualization
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black for noise
        col = 'k'
        marker = 'x'
        label = 'Noise'
    else:
        marker = 'o'
        label = f'Cluster {k}'
    
    class_member_mask = (labels == k)
    xy = X_scaled[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                s=100, label=label, alpha=0.6)

plt.title(f'DBSCAN Clustering (eps={0.3}, min_samples={5})')
plt.legend()
plt.show()
```

### 5.2 Parameter Tuning with K-Distance Graph

```python
from sklearn.neighbors import NearestNeighbors

# Compute k-nearest neighbors
k = 5  # min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sort distances to k-th nearest neighbor
k_distances = np.sort(distances[:, k-1], axis=0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.ylabel(f'{k}-NN Distance')
plt.xlabel('Data Points (sorted)')
plt.title('K-Distance Graph for Epsilon Selection')
plt.axhline(y=0.3, color='r', linestyle='--', label='Suggested ε = 0.3')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.3 Comparing DBSCAN with K-Means and GMM

```python
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Generate challenging dataset
from sklearn.datasets import make_moons
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', s=200, marker='X', edgecolors='black', linewidth=2)
axes[0].set_title('K-Means (Fails on non-convex)')

# GMM
gmm = GaussianMixture(n_components=2, random_state=42)
y_gmm = gmm.fit_predict(X_scaled)
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_gmm, cmap='viridis', s=50)
axes[1].set_title('GMM (Also fails)')

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)
axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='viridis', s=50)
axes[2].set_title('DBSCAN (Succeeds!)')

plt.tight_layout()
plt.show()
```

### 5.4 HDBSCAN (Hierarchical DBSCAN)

```python
import hdbscan

# HDBSCAN automatically selects epsilon
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
labels = clusterer.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(12, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='Spectral', s=50)
plt.title('HDBSCAN (Automatic ε selection)')
plt.colorbar(label='Cluster ID')
plt.show()

# Cluster membership strengths
plt.figure(figsize=(12, 4))
plt.bar(range(len(clusterer.probabilities_)), clusterer.probabilities_)
plt.xlabel('Point Index')
plt.ylabel('Membership Strength')
plt.title('Cluster Membership Probabilities')
plt.show()
```

---

## 6. Exam Preparation

### 6.1 Key Definitions Summary

| Term | Definition | Example |
|------|------------|---------|
| **Core Point** | Has $\geq$ MinPts in $\epsilon$-neighborhood | Downtown restaurant |
| **Border Point** | Not core, but in neighborhood of core | Suburban house near downtown |
| **Noise Point** | Neither core nor border | Isolated cabin in woods |
| **Density-Reachable** | Connected via chain of core points | Can walk downtown through crowds |

### 6.2 Algorithm Walkthrough Problem

**Given:** Points A, B, C, D, E with distances:
```
  d(A,B) = 1.0    d(A,C) = 0.5    d(A,D) = 3.0    d(A,E) = 4.0
  d(B,C) = 1.2    d(B,D) = 2.8    d(B,E) = 3.5
  d(C,D) = 2.5    d(C,E) = 3.2
  d(D,E) = 1.1
```

**Parameters:** $\epsilon = 1.5$, MinPts = 2

**Solution:**

**Step 1:** Find neighbors for each point
- $N_\epsilon(A) = \{B, C\}$ → $|N_\epsilon(A)| = 2$ → **Core**
- $N_\epsilon(B) = \{A, C\}$ → $|N_\epsilon(B)| = 2$ → **Core**
- $N_\epsilon(C) = \{A, B\}$ → $|N_\epsilon(C)| = 2$ → **Core**
- $N_\epsilon(D) = \{E\}$ → $|N_\epsilon(D)| = 1$ → **Not Core**
- $N_\epsilon(E) = \{D\}$ → $|N_\epsilon(E)| = 1$ → **Not Core**

**Step 2:** Determine point types
- A, B, C: Core points
- D: Border (in $N_\epsilon(E)$... wait, E is not core)
- E: Border (in $N_\epsilon(D)$... same issue)
- Actually, both D and E are **Noise** (not in neighborhood of any core point)

**Step 3:** Form clusters
- Cluster 1: {A, B, C} (all density-connected)
- Noise: {D, E}

### 6.3 Common Exam Questions

**Q1: Why is DBSCAN better than K-Means for arbitrarily shaped clusters?**

**Answer:**  
K-Means assumes clusters are convex (spherical) because it assigns points to nearest centroid. This creates Voronoi-like partitions with linear boundaries.

DBSCAN defines clusters by density connectivity, which can follow curved, elongated, or irregular boundaries. It only requires dense regions to be connected, not convex.

**Example:** For crescent-moon shaped clusters, K-Means will split them with a line, while DBSCAN follows the curve.

**Q2: What happens if ε is too small or too large?**

**ε too small:**
- Too many noise points
- True clusters fragmented into many small clusters
- Extreme: Every point is noise

**ε too large:**
- Clusters merge together
- Many distinct clusters become one
- Extreme: All points in one cluster

**Visual:**
```
True:           ε small:        ε large:
●●●  ◆◆◆        ● ● ●  ◆ ◆ ◆     ●●●◆◆◆
●●●  ◆◆◆        ● × ●  ◆ ◆ ◆     ●●●◆◆◆
●●●  ◆◆◆        ● ● ●  ◆ × ◆     ●●●◆◆◆
```

**Q3: Compare DBSCAN with K-Means and GMM.**

| Aspect | K-Means | GMM | DBSCAN |
|--------|---------|-----|--------|
| **Shape** | Spherical | Elliptical | Arbitrary |
| **Outliers** | Forces into clusters | Forced into clusters | Explicit noise |
| **$K$ required** | Yes | Yes | No |
| **Deterministic** | Yes (with seed) | No | Mostly yes |
| **Complexity** | $O(nKdI)$ | $O(nKd^2I)$ | $O(n^2)$ or $O(n \log n)$ |
| **Density** | Uniform assumed | Uniform assumed | Handles varying |

### 6.4 Interview Questions

**Technical:**

**Q:** You run DBSCAN and get 1 large cluster + many single-point "clusters" (noise). What's wrong?

**A:** Likely issues:
1. **ε too small:** Increase $\epsilon$ gradually
2. **MinPts too high:** Lower MinPts
3. **Need scaling:** Features have different scales (standardize!)
4. **Actually noisy data:** May be legitimate if dataset is truly sparse

**Diagnostic:** Plot k-distance graph to find appropriate $\epsilon$.

**Q:** DBSCAN is $O(n^2)$. How do we scale to millions of points?

**A:** Solutions:
1. **Spatial indexing:** Use kd-tree or ball-tree → $O(n \log n)$
2. **Sampling:** Run on random sample, then assign remaining points
3. **HDBSCAN:** Hierarchical variant often faster
4. **Approximate DBSCAN:** Trade accuracy for speed
5. **Grid-based preprocessing:** Quantize space into cells

**Practical:**

**Q:** When would you use DBSCAN in a real project?

**A:** Use DBSCAN when:
- **Geography/GIS:** Finding hotspots (crime, disease outbreak)
- **Anomaly Detection:** Financial fraud (normal = dense cluster, fraud = noise)
- **Network Analysis:** Community detection in social networks
- **Image Segmentation:** Objects have irregular shapes
- **Marketing:** Customer segments with unclear boundaries

**Don't use** when:
- Need exact number of clusters
- All data should be clustered (no noise allowed)
- Very high dimensions (distance becomes meaningless)
- Clusters have vastly different densities

---

## Summary

**Key Takeaways:**
1. DBSCAN uses density connectivity, not centroid distance
2. Automatically identifies clusters and noise
3. Can find arbitrarily shaped clusters
4. Key parameters: $\epsilon$ (radius), MinPts (density threshold)
5. Use k-distance graph for parameter selection

**Algorithm Steps:**
1. Classify points: Core, Border, Noise
2. Form clusters from density-connected core points
3. Assign border points to nearest cluster

**Strengths:**
- Handles non-convex shapes
- Robust to outliers
- No need to specify $K$
- Deterministic

**Weaknesses:**
- Parameter sensitivity
- Struggles with varying densities
- High-dimensional challenges
- $O(n^2)$ without indexing

**Practical Tips:**
- Always standardize features
- Use k-distance graph for $\epsilon$
- Set MinPts ≥ dimensionality + 1
- Try HDBSCAN for automatic tuning
- Visualize results to validate parameters
