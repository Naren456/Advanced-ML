# Class 08: Anomaly Detection - Machine Learning Methods

> **Core Principle:** "Leveraging algorithms to automatically identify outliers"

---

## Table of Contents
1. [Advanced Anomaly Detection](#1-advanced-anomaly-detection)
2. [Isolation Forest](#2-isolation-forest)
3. [Local Outlier Factor](#3-local-outlier-factor)
4. [One-Class SVM](#4-one-class-svm)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Advanced Anomaly Detection

### 1.1 Limitations of Statistical Methods

**Problems with IQR/Z-Score:**
1. **Univariate:** Analyze each feature independently
2. **Linear boundaries:** Miss complex patterns
3. **Global approach:** Assume uniform density
4. **No learning:** Don't adapt to data structure

**Example Failure Case:**
```
Feature 1 vs Feature 2:

    F2
    |
  5 |    Normal    × (outlier)
    |   cluster
  0 |     ●●●       
    |    ●●●●●      
 -5 |     ●●●       
    |_____________ F1
   -5   0   5

Point × : Normal in F1, normal in F2, but outlier jointly!
```

### 1.2 Machine Learning Approach

**Key Idea:** Use algorithms that learn data structure

**Advantages:**
- Handle multivariate data naturally
- Capture non-linear relationships
- Adapt to local density variations
- No distribution assumptions

---

## 2. Isolation Forest

### 2.1 Core Philosophy

**Fundamental Insight:** Anomalies are "few and different" → **easier to isolate**

**Analogy:** Finding a specific book in a library:
- **Popular book (normal):** Fiction section, 3rd shelf, 50th position → Many steps
- **Rare book (anomaly):** "Oh, that weird one? It's right here" → Few steps

### 2.2 The Isolation Tree

**Random Binary Tree:** Recursively partition data with random splits

**Construction Algorithm:**

```
BUILD_TREE(X, current_height, max_height):
    IF current_height >= max_height OR |X| <= 1:
        RETURN leaf node
    
    # Random split
    feature = RANDOM_SELECT(features)
    split_value = RANDOM_VALUE(min(X[feature]), max(X[feature]))
    
    X_left = X[X[feature] < split_value]
    X_right = X[X[feature] >= split_value]
    
    RETURN node with:
        left_child = BUILD_TREE(X_left, current_height+1, max_height)
        right_child = BUILD_TREE(X_right, current_height+1, max_height)
```

### 2.3 Path Length

**Path Length $h(x)$:** Number of edges from root to leaf containing $x$

**Key Observation:**
- **Normal points:** Deep in tree (long path)
- **Anomalies:** Shallow in tree (short path)

**Visual Example:**
```
         Root
        /    \
       /      \
      *        Normal
   Anomaly     Region
   (h=1)         |
                 |
              (more
               splits)
                 |
              (h=8)
```

### 2.4 Anomaly Score

**Average Path Length:** Build multiple trees ($t = 100-200$), average path lengths

$$h(x) = \text{average path length across all trees}$$

**Normalization:** Use average path length of unsuccessful search in BST

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

where $H(i) = \ln(i) + 0.5772$ (harmonic number)

**Anomaly Score:**

$$s(x, n) = 2^{-\frac{h(x)}{c(n)}}$$

**Interpretation:**
- $s \to 1$: **Anomaly** (path length much smaller than average)
- $s \to 0$: **Normal** (path length much larger than average)
- $s \approx 0.5$: **Borderline**

### 2.5 Mathematical Analysis

**Theorem:** For dataset of size $n$:
- Average path length for normal points: $\approx c(n)$
- Expected path length for point $x$: $E[h(x)]$

**Properties:**
1. **Sub-sampling:** Use $\psi = 256$ samples per tree (sufficient)
2. **Tree limit:** Typically $100$ trees
3. **Contamination:** Expected proportion of outliers (default: 0.1)

**Complexity:**
- **Training:** $O(t \cdot \psi \cdot \log \psi)$ where $t$ = number of trees
- **Testing:** $O(t \cdot \log \psi)$
- **Space:** $O(t \cdot \psi)$

---

## 3. Local Outlier Factor (LOF)

### 3.1 Motivation

**Problem:** Global methods fail with varying densities

**Example:**
```
Dense cluster A:     Sparse cluster B:
  ●●●●●               ●   ●
  ●●●●●               
  ●●●●●               ●   ●
  
Point in B looks like outlier globally but is normal locally!
```

### 3.2 Core Concepts

**Definition 1: k-distance**  
Distance to $k$-th nearest neighbor

$$d_k(A) = \text{distance to } k\text{-th nearest neighbor of } A$$

**Definition 2: Reachability Distance**

$$\text{reach-dist}_k(A, B) = \max\{d_k(B), d(A, B)\}$$

**Intuition:** Distance from $A$ to $B$, but never less than $B$'s k-distance (smoothing effect)

**Definition 3: Local Reachability Density (LRD)**

$$\text{lrd}_k(A) = \frac{1}{\frac{\sum_{B \in N_k(A)} \text{reach-dist}_k(A, B)}{|N_k(A)|}}$$

**Interpretation:** Inverse of average reachability distance → **density** around $A$

### 3.3 LOF Score

**Local Outlier Factor:**

$$\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}}{|N_k(A)|}$$

**Simplified:** Ratio of neighbor densities to point's density

**Interpretation:**
- $\text{LOF} \approx 1$: **Similar density to neighbors** (normal)
- $\text{LOF} \gg 1$: **Much lower density than neighbors** (outlier)
- $\text{LOF} < 1$: **Higher density than neighbors** (in dense core)

**Threshold:** Typically $\text{LOF} > 1.5$ or $2.0$ indicates outlier

### 3.4 Step-by-Step Example

**Given:** Points $A, B, C, D, E$ with $k=2$

**Step 1:** Compute k-distances
- $d_2(A) = 1.5$ (distance to 2nd nearest neighbor)
- $d_2(B) = 1.2$
- ...

**Step 2:** Compute reachability distances
- $\text{reach-dist}_2(A, B) = \max\{d_2(B), d(A,B)\} = \max\{1.2, 0.8\} = 1.2$

**Step 3:** Compute LRD
- $\text{lrd}_2(A) = \frac{1}{\text{avg reachability distance}}$

**Step 4:** Compute LOF
- $\text{LOF}_2(A) = \frac{\text{avg}(\text{lrd of neighbors})}{\text{lrd}(A)}$

---

## 4. One-Class SVM

### 4.1 Concept

**Goal:** Learn a boundary encompassing normal data

**Approach:** Find smallest hypersphere (or hyperplane) containing most points

**Mathematical Formulation:**

$$\min_{w, \rho, \xi} \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu n}\sum_{i=1}^{n} \xi_i$$

subject to:
$$w^T\phi(x_i) \geq \rho - \xi_i, \quad \xi_i \geq 0$$

where:
- $\phi(x)$: Feature mapping (kernel trick)
- $\rho$: Offset from origin
- $\nu \in (0,1)$: Upper bound on fraction of outliers
- $\xi_i$: Slack variables

### 4.2 Kernel Functions

**RBF (Radial Basis Function):**

$$K(x, y) = \exp\left(-\gamma \|x - y\|^2\right)$$

**Properties:**
- $\gamma$ controls smoothness
- High $\gamma$: Tight, complex boundary
- Low $\gamma$: Loose, simple boundary

### 4.3 Parameters

**$\nu$ (nu):** Expected fraction of outliers (0.01 to 0.5)

**$\gamma$:** RBF kernel coefficient (auto: $1/n_{features}$)

---

## 5. Implementation

### 5.1 Isolation Forest

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate data with outliers
np.random.seed(42)
X_normal = np.random.randn(300, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# Fit Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,        # Number of trees
    max_samples=256,         # Subsample size
    contamination=0.1,       # Expected outlier fraction
    random_state=42
)

# Predict (-1 for outliers, 1 for inliers)
y_pred = iso_forest.fit_predict(X)

# Anomaly scores (lower = more anomalous)
scores = iso_forest.decision_function(X)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Predictions
axes[0].scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], 
               c='blue', label='Normal', alpha=0.6, s=50)
axes[0].scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], 
               c='red', label='Anomaly', marker='x', s=200, linewidths=3)
axes[0].set_title('Isolation Forest Predictions')
axes[0].legend()

# Plot 2: Anomaly Scores
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=scores, 
                         cmap='RdYlGn', s=50, alpha=0.7)
axes[1].set_title('Anomaly Scores (Lower = More Anomalous)')
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.show()

# Statistics
print(f"Number of outliers detected: {(y_pred == -1).sum()}")
print(f"Outlier indices: {np.where(y_pred == -1)[0][:10]}")
```

### 5.2 Local Outlier Factor

```python
from sklearn.neighbors import LocalOutlierFactor

# Fit LOF
lof = LocalOutlierFactor(
    n_neighbors=20,         # Number of neighbors
    contamination=0.1,      # Expected outlier fraction
    novelty=False           # False: fit_predict, True: fit then predict
)

# Predict
y_pred_lof = lof.fit_predict(X)

# LOF scores (negative, more negative = more anomalous)
lof_scores = lof.negative_outlier_factor_

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Predictions
axes[0].scatter(X[y_pred_lof == 1, 0], X[y_pred_lof == 1, 1], 
               c='blue', label='Normal', alpha=0.6, s=50)
axes[0].scatter(X[y_pred_lof == -1, 0], X[y_pred_lof == -1, 1], 
               c='red', label='Anomaly', marker='x', s=200, linewidths=3)
axes[0].set_title('LOF Predictions')
axes[0].legend()

# Plot 2: LOF Scores
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=-lof_scores,  # Negate for visualization
                         cmap='RdYlGn_r', s=50, alpha=0.7)
axes[1].set_title('LOF Scores (Higher = More Anomalous)')
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.show()
```

### 5.3 One-Class SVM

```python
from sklearn.svm import OneClassSVM

# Fit One-Class SVM
oc_svm = OneClassSVM(
    kernel='rbf',          # Radial Basis Function
    gamma='auto',          # 1 / n_features
    nu=0.1                 # Upper bound on outliers
)

# Fit and predict
y_pred_svm = oc_svm.fit_predict(X)

# Decision function (distance to separating hyperplane)
svm_scores = oc_svm.decision_function(X)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Predictions with decision boundary
axes[0].scatter(X[y_pred_svm == 1, 0], X[y_pred_svm == 1, 1], 
               c='blue', label='Normal', alpha=0.6, s=50)
axes[0].scatter(X[y_pred_svm == -1, 0], X[y_pred_svm == -1, 1], 
               c='red', label='Anomaly', marker='x', s=200, linewidths=3)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
Z = oc_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[0].contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
axes[0].set_title('One-Class SVM with Decision Boundary')
axes[0].legend()

# Plot 2: Decision scores
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=svm_scores, 
                         cmap='RdYlGn', s=50, alpha=0.7)
axes[1].set_title('Decision Function Values')
plt.colorbar(scatter, ax=axes[1])

plt.tight_layout()
plt.show()
```

### 5.4 Comparison of Methods

```python
# Apply all methods
methods = {
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
    'LOF': LocalOutlierFactor(contamination=0.1),
    'One-Class SVM': OneClassSVM(nu=0.1)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(methods.items()):
    # Predict
    if name == 'LOF':
        y_pred = model.fit_predict(X)
    else:
        y_pred = model.fit(X).predict(X)
    
    # Plot
    axes[idx].scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], 
                     c='blue', alpha=0.6, s=50, label='Normal')
    axes[idx].scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], 
                     c='red', marker='x', s=200, linewidths=3, label='Anomaly')
    axes[idx].set_title(f'{name}\n({(y_pred == -1).sum()} outliers)')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6. Exam Preparation

### 6.1 Method Comparison Table

| Feature | Isolation Forest | LOF | One-Class SVM |
|---------|-----------------|-----|---------------|
| **Principle** | Isolation ease | Local density | Boundary separation |
| **Complexity** | $O(t n \log n)$ | $O(n^2)$ | $O(n^2)$ to $O(n^3)$ |
| **Scalability** | Excellent | Poor | Moderate |
| **Global/Local** | Global | Local | Global |
| **Parameters** | Few | Few | Several |
| **Interpretability** | High | Medium | Low |
| **Anomaly types** | Global | Local | Global |
| **Training data** | All data | All data | Normal only |

### 6.2 Key Formulas

**Isolation Forest:**

$$s(x, n) = 2^{-\frac{h(x)}{c(n)}}$$

where $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$

**LOF:**

$$\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \text{lrd}_k(B) / \text{lrd}_k(A)}{|N_k(A)|}$$

### 6.3 Common Exam Questions

**Q1: Why is Isolation Forest faster than LOF?**

**Answer:**  
**Isolation Forest:** $O(t \cdot n \cdot \log \psi)$
- Builds $t$ trees independently (parallelizable)
- Each tree uses small subsample ($\psi = 256$)
- Tree depth is $O(\log \psi)$

**LOF:** $O(n^2)$ or $O(kn \log n)$ with indexing
- Must compute distances between all point pairs
- For each point, find $k$ nearest neighbors
- Calculate densities for each point and neighbors

**For large $n$:** Isolation Forest scales much better.

**Q2: When would you use LOF instead of Isolation Forest?**

**Use LOF when:**
1. **Varying densities:** Data has clusters of different densities
2. **Local anomalies:** Outliers are local (normal in one region, anomalous in another)
3. **Small dataset:** $n < 10{,}000$ (LOF's quadratic complexity manageable)
4. **Interpretability:** Need to understand why point is anomalous (density-based explanation)

**Use Isolation Forest when:**
1. **Large dataset:** $n > 100{,}000$
2. **Global anomalies:** Outliers are clearly separated from bulk
3. **High dimensions:** Works well even with $d > 100$
4. **Speed critical:** Real-time anomaly detection

**Q3: Explain "contamination" parameter.**

**Answer:**  
**Contamination:** Expected proportion of outliers in dataset

**Usage:**
- Helps algorithm set decision threshold
- If contamination = 0.1, algorithm will mark top 10% most anomalous points as outliers

**Setting guidelines:**
- **Known outliers:** Use actual proportion (e.g., 0.05 = 5%)
- **Unknown:** Start with 0.1 (10%), tune based on results
- **Very clean data:** Use 0.01-0.05
- **Noisy data:** Use 0.1-0.2

**Warning:** Setting too high can label normal points as anomalies!

### 6.4 Interview Questions

**Technical:**

**Q:** You run Isolation Forest and LOF on same data, get different results. Which to trust?

**A:** Depends on data structure:

**Agreement:** Both methods agree → High confidence

**Disagreement:** Investigate!
1. **Visualize:** Plot both results
2. **Check density:** If varying densities → trust LOF
3. **Check separability:** If clear global outliers → trust Isolation Forest
4. **Domain expertise:** Which makes more sense?
5. **Ensemble:** Combine predictions (outlier if both say so)

**Best practice:** Use multiple methods as ensemble

**Q:** Isolation Forest marks 50% of data as anomalies. What's wrong?

**A:** Possible issues:
1. **Contamination too high:** Should match true outlier proportion (<< 50%)
2. **Wrong kernel/parameters:** For One-Class SVM
3. **Data needs scaling:** Features with different ranges
4. **Wrong algorithm choice:** Data structure doesn't match assumption
5. **Actual data quality issue:** Maybe data IS mostly anomalous!

**Diagnostic steps:**
- Check contamination parameter
- Visualize data distribution
- Try different methods
- Consult domain experts

**Practical:**

**Q:** You're building fraud detection for credit cards. Which method?

**A:** **Isolation Forest** is preferred because:

1. **Scalability:** Millions of transactions per day
2. **Speed:** Real-time detection required
3. **High dimensions:** Many features (amount, time, location, merchant, history)
4. **Ensemble-friendly:** Can combine with other models

**However:** Also use LOF for specific scenarios:
- Different customer segments have different spending patterns (varying densities)
- Local anomalies (unusual for *this* customer)

**Best solution:** Hybrid approach
- Isolation Forest for fast initial screening
- LOF for detailed analysis of flagged transactions
- Supervised classifier if labeled fraud data available

---

## Summary

**Key Takeaways:**
1. Machine learning methods handle complex, high-dimensional data
2. Isolation Forest: Fast, scalable, isolation-based
3. LOF: Handles varying densities, local anomalies
4. One-Class SVM: Learns decision boundary
5. Choose method based on data structure and requirements

**Method Selection Guide:**
- **Large dataset + speed:** Isolation Forest
- **Varying densities:** LOF
- **Need boundary:** One-Class SVM
- **Unsure:** Try multiple, ensemble

**Best Practices:**
- Always scale/normalize features
- Set contamination parameter carefully
- Visualize results
- Validate with domain expertise
- Use ensemble of multiple methods
- Monitor performance over time

**Common Applications:**
- Fraud detection
- Network intrusion
- Manufacturing defects
- System health monitoring
- Data quality assessment
