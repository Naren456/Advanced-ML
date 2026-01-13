# Class 05: Gaussian Mixture Models (GMM)

> **Core Principle:** "Probabilistic clustering with soft assignments"

---

## Table of Contents
1. [Limitations of Hard Clustering](#1-limitations-of-hard-clustering)
2. [Gaussian Distribution Review](#2-gaussian-distribution-review)
3. [Gaussian Mixture Models Theory](#3-gaussian-mixture-models-theory)
4. [Expectation-Maximization Algorithm](#4-expectation-maximization-algorithm)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Limitations of Hard Clustering

### 1.1 The Binary Decision Problem

**K-Means Limitation:** Every point assigned to exactly one cluster

**Analogy:** Imagine classifying music genres. A song that blends jazz and classical (fusion) must be forced into one category, losing nuance.

```
Hard Clustering (K-Means):         Soft Clustering (GMM):
      ●●●  |  ◆◆◆                    ●●●~~~~◆◆◆
      ●●●  |  ◆◆◆                    ●●●~~~~◆◆◆
      ●●●  |  ◆◆◆                    ●●●~~~~◆◆◆
  (Sharp boundary)              (Gradual transition)
```

### 1.2 Need for Probabilistic Framework

**Scenario:** Point $x$ is equidistant from two centroids

**K-Means:** Arbitrary assignment (whichever is encountered first)

**GMM:** $P(x \in \text{Cluster 1}) = 0.5$, $P(x \in \text{Cluster 2}) = 0.5$


### 1.3 Real-World Analogies

**Example 1: Customer Segmentation (E-commerce)**
Consider classifying customers based on "Amount Spent" and "Number of Views":
- **Spenders:** High spend, low views (wealthy but decided)
- **Discount Seekers:** Low spend, frequent views (price conscious)
- **Explorers:** Low spend, high views (window shoppers)
*Reality:* A "Discount Seeker" might occasionally splurge like a "Spender". GMM captures this by assigning a 10% probability of being a Spender, rather than a hard "No".

**Example 2: Population Distributions (Hair Length)**
If you measure hair length of all citizens:
- **Unimodal:** Measuring only men might give a single bell curve (short hair).
- **Bimodal:** Measuring everyone gives two peaks (short for men, long for women).
- GMM can model this "Mixture" of two underlying Gaussian distributions.

---

## 2. Gaussian Distribution Review

### 2.1 Univariate Gaussian

**Probability Density Function:**

$$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Parameters:**
- $\mu$: mean (location)
- $\sigma^2$: variance (spread)

**Visual:**
```
  pdf
   |     
0.4|       ●
   |      ●●●
0.2|     ●●●●●
   |   ●●●●●●●●●
  0|_●●●●●●●●●●●●●_ x
    -3  -1  μ  1  3
    
    ←─ σ ─→
```

### 2.2 Multivariate Gaussian

**For $d$-dimensional data:**

$$\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Parameters:**
- $\boldsymbol{\mu} \in \mathbb{R}^d$: mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$: covariance matrix

**2D Example:**

$$\boldsymbol{\Sigma} = \begin{bmatrix} \sigma_x^2 & \rho\sigma_x\sigma_y \\ \rho\sigma_x\sigma_y & \sigma_y^2 \end{bmatrix}$$

where $\rho$ is correlation coefficient.

### 2.3 Covariance Matrix Types

**Spherical ($\sigma^2 \mathbf{I}$):**
```
    y
    |   ●●●
    |  ●●●●●
    |   ●●●
    |________ x
  (Circular)
```

**Diagonal:**
```
    y
    | ●●●●●●●
    | ●●●●●●●
    | ●●●●●●●
    |________ x
  (Axis-aligned ellipse)
```

**Full:**
```
    y
    |  ●●●●
    | ●●●●●●
    |●●●●●●●
    |________ x
  (Arbitrary orientation ellipse)
```

---

## 3. Gaussian Mixture Models Theory

### 3.1 Core Assumption

**Model:** Data is generated from $K$ Gaussian distributions

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where:
- $\pi_k$: mixing coefficient (prior probability of cluster $k$)
- $\sum_{k=1}^{K} \pi_k = 1$, $\pi_k \geq 0$

**Interpretation:** Each data point is drawn from one of $K$ Gaussians, but we don't know which one.

### 3.2 Latent Variable Model

**Introduce:** Hidden variable $z_i \in \{1, \ldots, K\}$ indicating which Gaussian generated $x_i$

**Joint Distribution:**

$$p(\mathbf{x}_i, z_i = k) = \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Marginal:**

$$p(\mathbf{x}_i) = \sum_{k=1}^{K} p(\mathbf{x}_i, z_i = k)$$

### 3.3 Responsibility (Posterior Probability)

**Question:** Given observation $\mathbf{x}_i$, what's the probability it belongs to cluster $k$?

**Bayes' Rule:**

$$\gamma_{ik} = p(z_i = k | \mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**Interpretation:** $\gamma_{ik}$ is the "responsibility" of cluster $k$ for point $i$

**Example:**
- $\gamma_{i1} = 0.8$, $\gamma_{i2} = 0.15$, $\gamma_{i3} = 0.05$
- Point $i$ primarily belongs to cluster 1, with small contributions from 2 and 3

---

## 4. Expectation-Maximization Algorithm

### 4.1 The Challenge

**Problem:** We have a "chicken-and-egg" situation:
- To estimate parameters $(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k)$, we need to know cluster assignments
- To compute assignments ($\gamma_{ik}$), we need parameters

**Solution:** EM algorithm - iteratively refine both

### 4.2 EM Algorithm Steps

**Initialization:**
Randomly initialize $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$, $\pi_k$ for $k = 1, \ldots, K$

**Repeat until convergence:**

**E-Step (Expectation):**  
Compute responsibilities given current parameters:

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M-Step (Maximization):**  
Update parameters given responsibilities:

$$N_k = \sum_{i=1}^{n} \gamma_{ik}$$

$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} \mathbf{x}_i$$

$$\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

$$\pi_k = \frac{N_k}{n}$$

**Convergence:** When log-likelihood change $< \epsilon$

### 4.3 Log-Likelihood

**Objective Function (what EM maximizes):**

$$\log p(\mathbf{X} | \boldsymbol{\theta}) = \sum_{i=1}^{n} \log \left(\sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right)$$

**Theorem:** EM is guaranteed to increase log-likelihood at each iteration (or keep it constant).

### 4.4 Intuitive Explanation of E-Step and M-Step

**E-Step Analogy:** "Soft labeling"  
- Like grading an exam with partial credit
- Instead of assigning point to one cluster (hard 0/1), assign probabilities

**M-Step Analogy:** "Weighted averaging"  
- Like computing class average where homework counts 30%, midterm 30%, final 40%
- Update each cluster's parameters using weighted contributions from all points

---

## 5. Implementation

### 5.1 Basic GMM

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit GMM
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)

# Predictions
y_pred = gmm.predict(X)           # Hard assignment
proba = gmm.predict_proba(X)      # Soft assignment (probabilities)

# Results
print("Means:")
print(gmm.means_)
print("\nCovariances shape:", gmm.covariances_.shape)
print("Weights:", gmm.weights_)
print("\nExample probabilities for first point:", proba[0])
```

### 5.2 Covariance Types

```python
covariance_types = ['spherical', 'diag', 'tied', 'full']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, cov_type in enumerate(covariance_types):
    ax = axes[idx // 2, idx % 2]
    
    gmm = GaussianMixture(n_components=4, covariance_type=cov_type, random_state=42)
    y_pred = gmm.fit_predict(X)
    
    # Plot data
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.6)
   
    # Plot Gaussian ellipses
    for k in range(4):
        mean = gmm.means_[k]
        covar = gmm.covariances_[k] if cov_type == 'full' else np.diag(gmm.covariances_[k]) if cov_type == 'diag' else gmm.covariances_ * np.eye(2) if cov_type == 'tied' else gmm.covariances_[k] * np.eye(2)
        
        # Eigenvalues for ellipse
        vals, vecs = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals) * 2  # 2-sigma ellipse
        
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(ellipse)
    
    ax.set_title(f'Covariance Type: {cov_type}', fontsize=12)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### 5.3 Model Selection with BIC/AIC

```python
from sklearn.mixture import GaussianMixture

# Test different number of components
n_components_range = range(1, 10)
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(n_components_range, bic_scores, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Components')
ax1.set_ylabel('BIC')
ax1.set_title('Bayesian Information Criterion')
ax1.grid(True, alpha=0.3)

ax2.plot(n_components_range, aic_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('AIC')
ax2.set_title('Akaike Information Criterion')
ax2.grid(True, alpha=0.3)

# Optimal K
optimal_bic = n_components_range[np.argmin(bic_scores)]
optimal_aic = n_components_range[np.argmin(aic_scores)]
print(f"Optimal components (BIC): {optimal_bic}")
print(f"Optimal components (AIC): {optimal_aic}")

plt.tight_layout()
plt.show()
```

### 5.4 Anomaly Detection with GMM

```python
# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Compute log-likelihood for each point
log_likelihood = gmm.score_samples(X)

# Anomalies are points with very low likelihood
threshold = np.percentile(log_likelihood, 5)  # Bottom 5%
anomalies = log_likelihood < threshold

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', label='Normal', alpha=0.6)
plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', label='Anomaly', s=100, marker='x')
plt.legend()
plt.title('Anomaly Detection with GMM')
plt.show()
```

---

## 6. Exam Preparation

### 6.1 Key Formulas Summary

| Concept | Formula | Notes |
|---------|---------|-------|
| Mixture Model | $p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \| \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | Weighted sum of Gaussians |
| Responsibility | $\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \| \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \| \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$ | Posterior probability |
| Mean Update | $\boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}$ | Weighted average |
| Covariance Update | $\boldsymbol{\Sigma}_k = \frac{\sum_i \gamma_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_i \gamma_{ik}}$ | Weighted covariance |

### 6.2 Numerical Example

**Given:** 1D data $\{1, 2, 9, 10\}$, $K=2$

**Initialize:**
- $\mu_1 = 1.5$, $\sigma_1^2 = 1$, $\pi_1 = 0.5$
- $\mu_2 = 9.5$, $\sigma_2^2 = 1$, $\pi_2 = 0.5$

**E-Step:** Compute $\gamma_{ik}$ for point $x = 2$

$$\gamma_{11} = \frac{0.5 \cdot \mathcal{N}(2 | 1.5, 1)}{0.5 \cdot \mathcal{N}(2 | 1.5, 1) + 0.5 \cdot \mathcal{N}(2 | 9.5, 1)}$$

$$\mathcal{N}(2 | 1.5, 1) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(2-1.5)^2}{2}\right) = 0.352$$

$$\mathcal{N}(2 | 9.5, 1) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(2-9.5)^2}{2}\right) \approx 5.55 \times 10^{-13}$$

$$\gamma_{11} = \frac{0.5 \cdot 0.352}{0.5 \cdot 0.352 + 0.5 \cdot 5.55 \times 10^{-13}} \approx 1.0$$

**M-Step:** Update $\mu_1$ using all responsibilities

$$N_1 = \sum_{i=1}^{4} \gamma_{i1} \approx 2$$

$$\mu_1 = \frac{\gamma_{11} \cdot 1 + \gamma_{21} \cdot 2 + \gamma_{31} \cdot 9 + \gamma_{41} \cdot 10}{N_1} \approx \frac{1 + 2}{2} = 1.5$$

### 6.3 Common Exam Questions

**Q1: Compare K-Means and GMM.**

| Aspect | K-Means | GMM |
|--------|---------|-----|
| **Assignment** | Hard (0 or 1) | Soft (probabilities) |
| **Model** | Non-probabilistic | Probabilistic |
| **Cluster Shape** | Spherical | Elliptical |
| **Distance Metric** | Euclidean | Mahalanobis |
| **Output** | Cluster labels | Probabilities |
| **Complexity** | Lower | Higher |
| **Parameters** | Centroids | Means, covariances, weights |
| **Generative** | No | Yes (can sample new data) |

**Q2: What are the different covariance types in GMM?**

1. **Spherical:** $\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$
   - Each cluster is spherical with same variance in all dimensions
   - Fewest parameters

2. **Diagonal:** $\boldsymbol{\Sigma}_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kd}^2)$
   - Axis-aligned ellipsoids
   - Different variance per dimension

3. **Tied:** $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ (same for all $k$)
   - All clusters have same shape/orientation
   - Saves parameters

4. **Full:** $\boldsymbol{\Sigma}_k$ is any positive definite matrix
   - Most flexible, arbitrary ellipsoids
   - Most parameters ($K \cdot \frac{d(d+1)}{2}$)

**Q3: How to choose the number of components $K$?**

**Methods:**
1. **BIC (Bayesian Information Criterion):**
   $$\text{BIC} = -2 \log \mathcal{L} + p \log n$$
   where $p$ is number of parameters, $n$ is sample size
   - **Lower is better**
   - Penalizes model complexity

2. **AIC (Akaike Information Criterion):**
   $$\text{AIC} = -2 \log \mathcal{L} + 2p$$
   - **Lower is better**
   - Less penalty than BIC

3. **Silhouette Score:** Same as K-Means

4. **Domain Knowledge**

### 6.4 Interview Questions

**Technical (ML Engineer):**

**Q:** When would you use GMM over K-Means?

**A:** Use GMM when:
1. **Soft assignments needed:** e.g., document topic modeling where a document can belong to multiple topics
2. **Non-spherical clusters:** GMM captures elliptical shapes
3. **Probabilistic interpretation required:** e.g., confidence scores, anomaly detection
4. **Generative modeling:** Need to sample new data points
5. **Statistical testing:** Want to model data distribution formally

**Q:** GMM is slow on large datasets. How can we speed it up?

**A:** Solutions:
1. **Mini-batch EM:** Update parameters on random subsets
2. **K-Means initialization:** Use K-Means++ results as starting point
3. **Reduce components:** Use BIC/AIC to find minimal $K$
4. **Simpler covariance:** Use 'spherical' or 'diag' instead of 'full'
5. **Parallel processing:** EM can be parallelized across data points

**Practical (Data Scientist):**

**Q:** You fit GMM and some cluster weights are very small (< 0.01). What does this mean?

**A:** 
- **Interpretation:** That component captures very few points (possibly outliers or noise)
- **Actions:**
  - Try reducing $K$ (may be overfitting)
  - Check if it's capturing a genuine rare subpopulation (domain knowledge)
  - Add regularization or use prior on weights (Bayesian GMM)

---

## Summary

**Key Takeaways:**
1. GMM is probabilistic clustering with soft assignments
2. Assumes data generated from mixture of Gaussians
3. EM algorithm iteratively refines parameters
4. Covariance type controls cluster shapes
5. BIC/AIC for model selection

**Advantages over K-Means:**
- Soft clustering (probabilities)
- Elliptical clusters
- Probabilistic framework
- Generative model
- Uncertainty quantification

**Limitations:**
- Slower than K-Means
- More parameters to estimate
- Still assumes Gaussian distributions
- Sensitive to initialization
- Convergence to local optimum

**Best Practices:**
- Use multiple random initializations
- Select covariance type based on data
- Validate with BIC/AIC scores
- Check convergence criteria
- Visualize clusters when possible
