# Class 07: Anomaly Detection - Statistical Methods

> **Core Principle:** "Identifying rare items that differ significantly from the majority"

---

## Table of Contents
1. [Introduction to Anomaly Detection](#1-introduction-to-anomaly-detection)
2. [Statistical Methods](#2-statistical-methods)
3. [IQR Method](#3-iqr-method)
4. [Z-Score Method](#4-z-score-method)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Introduction to Anomaly Detection

### 1.1 Problem Definition

**Anomaly (Outlier):** A data point that deviates significantly from the majority of the data

**Analogy:** Finding the one black sheep in a flock of 1000 white sheep

### 1.2 Types of Anomalies

**1. Point Anomalies**  
Individual instances that are anomalous

**Example:** Credit card transaction of $50,000 when typical is $50

**2. Contextual Anomalies**  
Anomalous in a specific context

**Example:** Temperature of 25°C is normal in summer but anomalous in winter

**3. Collective Anomalies**  
Collection of points is anomalous together

**Example:** Sequence of failed login attempts (each alone is normal)

### 1.3 Applications

| Domain | Application | What's Anomalous |
|--------|-------------|------------------|
| **Finance** | Fraud detection | Unusual transactions |
| **Healthcare** | Disease outbreaks | Spike in symptoms |
| **Manufacturing** | Quality control | Defective products |
| **Network Security** | Intrusion detection | Unusual traffic patterns |
| **IoT** | Sensor monitoring | Sensor malfunction |

### 1.4 Challenges

**Challenge 1: Imbalanced Data**  
Anomalies are rare (typically < 1% of data)

**Challenge 2: No Labels**  
Often unsupervised (don't know what anomalies look like)

**Challenge 3: Evolving Patterns**  
What's normal changes over time

**Challenge 4: High Dimensionality**  
Curse of dimensionality affects distance-based methods

---

## 2. Statistical Methods

### 2.1 Fundamental Assumption

**Parametric Approach:** Assume data follows a known distribution (typically Gaussian)

**Key Idea:** Points far from the distribution center are anomalous

### 2.2 Visual Intuition

```
Normal Distribution:

  Frequency
     |        Normal
     |        Region
     |      ╱──────╲
     |     ╱        ╲
     |    ╱          ╲
     |   ╱            ╲
     |__╱______________╲____ Value
     ↑               ↑
  Anomaly        Anomaly
  (too low)      (too high)
```

### 2.3 Comparison of Methods

| Method | Assumes | Robust to Outliers | Threshold | Best For |
|--------|---------|-------------------|-----------|----------|
| **IQR** | None | Yes | Q3 + 1.5×IQR | Skewed data |
| **Z-Score** | Normal distribution | No | \|z\| > 3 | Gaussian data |
| **Modified Z-Score** | Symmetric | Somewhat | \|z\| > 3.5 | Moderate outliers |

---

## 3. IQR Method

### 3.1 Theory

**Interquartile Range (IQR):** The range containing the middle 50% of data

**Mathematical Definition:**

$$\text{IQR} = Q_3 - Q_1$$

where:
- $Q_1$: 25th percentile (first quartile)
- $Q_3$: 75th percentile (third quartile)

### 3.2 Box Plot Connection

```
          Outlier
            ×
            
    ┌───────────────────┐
    │                   │
────┤       ┌───────┐   ├──── Upper Fence = Q3 + 1.5×IQR
    │       │       │   │
    │       │  Box  │   │
────┤       │       │   ├──── Median (Q2)
    │       │       │   │
    │       │       │   │
────┤       └───────┘   ├──── Lower Fence = Q1 - 1.5×IQR
    │                   │
    └───────────────────┘
    
            ×
          Outlier
```

### 3.3 Outlier Detection Rules

**Lower Fence:**

$$L = Q_1 - 1.5 \times \text{IQR}$$

**Upper Fence:**

$$U = Q_3 + 1.5 \times \text{IQR}$$

**Decision Rule:**

$$x \text{ is outlier if } x < L \text{ or } x > U$$

### 3.4 Why 1.5?

**Tukey's Original Justification:**  
For normal distribution, approximately 99.3% of data falls within $Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$

**Alternative Multipliers:**
- **1.5:** Standard (identifies "outliers")
- **3.0:** Conservative (identifies "extreme outliers")
- **1.0:** Aggressive (more false positives)

### 3.5 Step-by-Step Example

**Dataset:** $\{12, 15, 18, 20, 22, 25, 28, 30, 32, 100\}$

**Step 1:** Sort data (already sorted)

**Step 2:** Find quartiles
- $Q_1 = 18$ (25th percentile)
- $Q_2 = 23.5$ (median)
- $Q_3 = 30$ (75th percentile)

**Step 3:** Compute IQR

$$\text{IQR} = Q_3 - Q_1 = 30 - 18 = 12$$

**Step 4:** Calculate fences

$$L = 18 - 1.5 \times 12 = 18 - 18 = 0$$

$$U = 30 + 1.5 \times 12 = 30 + 18 = 48$$

**Step 5:** Identify outliers

$$\text{Outliers} = \{x : x < 0 \text{ or } x > 48\} = \{100\}$$

---

## 4. Z-Score Method

### 4.1 Theory

**Assumption:** Data follows normal distribution $\mathcal{N}(\mu, \sigma^2)$

**Z-Score Definition:**

$$z = \frac{x - \mu}{\sigma}$$

**Interpretation:** Number of standard deviations from the mean

### 4.2 Normal Distribution Properties

**Empirical Rule (68-95-99.7):**

```
  Probability
     |
     |     68%
     |   ╱────╲
     |  ╱      ╲
     | ╱  95%   ╲
     |╱          ╲
     ╱────────────╲
    ╱      99.7%   ╲
___╱________________╲___ z-score
  -3  -2  -1  0  1  2  3
```

- $\mu \pm 1\sigma$: Contains 68% of data
- $\mu \pm 2\sigma$: Contains 95% of data  
- $\mu \pm 3\sigma$: Contains 99.7% of data

### 4.3 Outlier Detection Rules

**Common Thresholds:**

| Threshold | Probability | Usage |
|-----------|-------------|-------|
| $\|z\| > 2$ | ~5% | Liberal (more outliers) |
| $\|z\| > 2.5$ | ~1.2% | Moderate |
| $\|z\| > 3$ | ~0.3% | Standard (most common) |
| $\|z\| > 3.5$ | ~0.05% | Conservative |

**Decision Rule (Standard):**

$$x \text{ is outlier if } |z| > 3$$

### 4.4 Modified Z-Score (Robust Variant)

**Problem:** Mean and std are sensitive to outliers

**Solution:** Use median and MAD (Median Absolute Deviation)

**Modified Z-Score:**

$$M_i = \frac{0.6745(x_i - \tilde{x})}{\text{MAD}}$$

where:
- $\tilde{x}$: median
- $\text{MAD} = \text{median}(|x_i - \tilde{x}|)$
- 0.6745: Constant making MAD consistent with $\sigma$ for normal data

**Threshold:** $|M_i| > 3.5$

### 4.5 Step-by-Step Example

**Dataset:** $\{12, 15, 18, 20, 22, 25, 28, 30, 32, 100\}$

**Step 1:** Calculate statistics

$$\mu = \frac{12 + 15 + \cdots + 100}{10} = 30.2$$

$$\sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{n}} = 25.64$$

**Step 2:** Compute z-scores

For $x = 100$:

$$z = \frac{100 - 30.2}{25.64} = 2.72$$

For $x = 12$:

$$z = \frac{12 - 30.2}{25.64} = -0.71$$

**Step 3:** Apply threshold ($|z| > 3$)

Since $|2.72| < 3$, point 100 is **not** an outlier by standard z-score!

**Note:** This illustrates z-score's weakness - the outlier itself inflates $\sigma$

**Modified Z-Score Approach:**

Median $\tilde{x} = 23.5$

$\text{MAD} = \text{median}(|12-23.5|, |15-23.5|, \ldots) = 7$

$$M_{100} = \frac{0.6745(100 - 23.5)}{7} = 7.36$$

Since $7.36 > 3.5$, point 100 **is** an outlier!

---

## 5. Implementation

### 5.1 IQR Method Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    data : array-like
        Input data
    multiplier : float
        IQR multiplier (default: 1.5)
    
    Returns:
    --------
    outliers : array
        Boolean mask of outliers
    bounds : tuple
        (lower_fence, upper_fence)
    """
    # Calculate quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Calculate fences
    lower_fence = Q1 - multiplier * IQR
    upper_fence = Q3 + multiplier * IQR
    
    # Identify outliers
    outliers = (data < lower_fence) | (data > upper_fence)
    
    return outliers, (lower_fence, upper_fence)

# Example usage
np.random.seed(42)
data = np.concatenate([np.random.normal(50, 10, 100), [5, 95, 100]])

outliers, (lower, upper) = detect_outliers_iqr(data)

print(f"Lower fence: {lower:.2f}")
print(f"Upper fence: {upper:.2f}")
print(f"Number of outliers: {outliers.sum()}")
print(f"Outlier values: {data[outliers]}")

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.boxplot(data, vert=False)
plt.axvline(lower, color='r', linestyle='--', label='Lower fence')
plt.axvline(upper, color='r', linestyle='--', label='Upper fence')
plt.xlabel('Value')
plt.title('Box Plot with Fences')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(data)), data, c=outliers, cmap='RdYlGn_r', s=50)
plt.axhline(lower, color='r', linestyle='--', alpha=0.5)
plt.axhline(upper, color='r', linestyle='--', alpha=0.5)
plt.ylabel('Value')
plt.xlabel('Index')
plt.title('Outlier Detection (IQR)')
plt.colorbar(label='Is Outlier')

plt.tight_layout()
plt.show()
```

### 5.2 Z-Score Method Implementation

```python
def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using Z-score method
    
    Parameters:
    -----------
    data : array-like
        Input data
    threshold : float
        Z-score threshold (default: 3)
    
    Returns:
    --------
    outliers : array
        Boolean mask of outliers
    z_scores : array
        Z-scores for all points
    """
    # Calculate mean and std
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate z-scores
    z_scores = (data - mean) / std
    
    # Identify outliers
    outliers = np.abs(z_scores) > threshold
    
    return outliers, z_scores

# Example
outliers_z, z_scores = detect_outliers_zscore(data, threshold=3)

print(f"Mean: {np.mean(data):.2f}")
print(f"Std: {np.std(data):.2f}")
print(f"Number of outliers: {outliers_z.sum()}")

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(z_scores, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(-3, color='r', linestyle='--', label='Threshold = ±3')
plt.axvline(3, color='r', linestyle='--')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.title('Distribution of Z-Scores')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(data)), z_scores, c=outliers_z, cmap='RdYlGn_r', s=50)
plt.axhline(-3, color='r', linestyle='--', alpha=0.5)
plt.axhline(3, color='r', linestyle='--', alpha=0.5)
plt.ylabel('Z-Score')
plt.xlabel('Index')
plt.title('Z-Score Outlier Detection')
plt.colorbar(label='Is Outlier')

plt.tight_layout()
plt.show()
```

### 5.3 Modified Z-Score (Robust)

```python
def detect_outliers_modified_zscore(data, threshold=3.5):
    """
    Detect outliers using Modified Z-score (robust to outliers)
    
    Parameters:
    -----------
    data : array-like
        Input data
    threshold : float
        Modified Z-score threshold (default: 3.5)
    
    Returns:
    --------
    outliers : array
        Boolean mask of outliers
    modified_z_scores : array
        Modified Z-scores for all points
    """
    # Calculate median
    median = np.median(data)
    
    # Calculate MAD (Median Absolute Deviation)
    mad = np.median(np.abs(data - median))
    
    # Calculate modified z-scores
    modified_z_scores = 0.6745 * (data - median) / mad
    
    # Identify outliers
    outliers = np.abs(modified_z_scores) > threshold
    
    return outliers, modified_z_scores

# Example
outliers_mod, mod_z_scores = detect_outliers_modified_zscore(data)

print(f"Median: {np.median(data):.2f}")
print(f"MAD: {np.median(np.abs(data - np.median(data))):.2f}")
print(f"Number of outliers: {outliers_mod.sum()}")

# Compare all methods
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

methods = [
    ('IQR', outliers, 'IQR Method'),
    ('Z-Score', outliers_z, 'Z-Score Method'),
    ('Modified Z', outliers_mod, 'Modified Z-Score')
]

for idx, (name, mask, title) in enumerate(methods):
    axes[idx].scatter(range(len(data)), data, c=mask, cmap='RdYlGn_r', s=50)
    axes[idx].set_title(f'{title}\n({mask.sum()} outliers)')
    axes[idx].set_xlabel('Index')
    axes[idx].set_ylabel('Value')

plt.tight_layout()
plt.show()
```

### 5.4 Multivariate Outlier Detection

```python
from scipy.spatial.distance import mahalanobis

def detect_outliers_mahalanobis(data, threshold=3):
    """
    Detect multivariate outliers using Mahalanobis distance
    
    Parameters:
    -----------
    data : array (n_samples, n_features)
        Input data
    threshold : float
        Chi-square threshold
    
    Returns:
    --------
    outliers : array
        Boolean mask of outliers
    distances : array
        Mahalanobis distances
    """
    # Calculate mean and covariance
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    cov_inv = np.linalg.inv(cov)
    
    # Calculate Mahalanobis distance for each point
    distances = np.array([
        mahalanobis(x, mean, cov_inv) for x in data
    ])
    
    # Threshold based on chi-square distribution
    # For threshold=3, using chi2.ppf(0.99, df=n_features)
    outliers = distances > threshold
    
    return outliers, distances

# Example with 2D data
np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
X_outliers = np.array([[5, 5], [-5, -5], [5, -5]])
X = np.vstack([X_normal, X_outliers])

outliers_mah, mah_dist = detect_outliers_mahalanobis(X, threshold=3)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(X[~outliers_mah, 0], X[~outliers_mah, 1], 
           c='blue', label='Normal', alpha=0.6)
plt.scatter(X[outliers_mah, 0], X[outliers_mah, 1], 
           c='red', s=200, marker='x', label='Outlier', linewidths=3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multivariate Outlier Detection (Mahalanobis)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. Exam Preparation

### 6.1 Key Formulas Summary

| Method | Formula | Threshold | Assumptions |
|--------|---------|-----------|-------------|
| **IQR** | $L = Q_1 - 1.5 \times IQR$<br>$U = Q_3 + 1.5 \times IQR$ | $x < L$ or $x > U$ | None |
| **Z-Score** | $z = \frac{x - \mu}{\sigma}$ | $\|z\| > 3$ | Normal distribution |
| **Modified Z** | $M = \frac{0.6745(x - \tilde{x})}{\text{MAD}}$ | $\|M\| > 3.5$ | Symmetric |
| **Mahalanobis** | $D = \sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}$ | $D > \chi^2_{df,\alpha}$ | Multivariate normal |

### 6.2 Worked Example

**Problem:** Given data $\{5, 12, 15, 18, 20, 22, 25, 28, 30, 85\}$, identify outliers using both IQR and Z-score methods.

**Solution:**

**IQR Method:**

**Step 1:** Find quartiles
- Sorted: already sorted
- $Q_1 = 15$ (between 3rd and 4th values)
- $Q_3 = 28$ (between 7th and 8th values)

**Step 2:** Calculate IQR

$$\text{IQR} = 28 - 15 = 13$$

**Step 3:** Calculate fences

$$L = 15 - 1.5(13) = 15 - 19.5 = -4.5$$
$$U = 28 + 1.5(13) = 28 + 19.5 = 47.5$$

**Step 4:** Identify outliers

$$\text{Outliers} = \{5 < -4.5? \text{ No}, 85 > 47.5? \text{ Yes}\} = \{85\}$$

**Z-Score Method:**

**Step 1:** Calculate statistics

$$\mu = \frac{5+12+\cdots+85}{10} = 26$$

$$\sigma = \sqrt{\frac{(5-26)^2 + (12-26)^2 + \cdots + (85-26)^2}{10}} = 22.52$$

**Step 2:** Calculate z-scores

For $x = 85$: $z = \frac{85-26}{22.52} = 2.62$

For $x = 5$: $z = \frac{5-26}{22.52} = -0.93$

**Step 3:** Apply threshold ($|z| > 3$)

No outliers detected! (Because outlier inflated $\sigma$)

**Comparison:** IQR correctly identifies 85 as outlier, Z-score fails due to masking effect.

### 6.3 Common Exam Questions

**Q1: Why is IQR preferred over Z-score for skewed distributions?**

**Answer:**  
IQR is based on quartiles (order statistics), which are robust to:
1. **Extreme values:** One outlier doesn't shift $Q_1$ or $Q_3$ significantly
2. **Skewness:** Works for any distribution shape
3. **No assumptions:** Doesn't require normality

Z-score assumes normal distribution and uses mean/std, which are heavily influenced by outliers and skewness.

**Q2: What is the "masking effect" in outlier detection?**

**Answer:**  
The masking effect occurs when outliers inflate the variance estimate, making them appear less extreme.

**Example:** In $\{1, 2, 3, 4, 100\}$:
- True std (without 100): ~1.3
- Actual std (with 100): ~39.4
- Z-score of 100: Only 2.5 (below threshold!)

**Solution:** Use robust methods (IQR, Modified Z-score, or iterative approaches)

**Q3: Compare univariate and multivariate outlier detection.**

**Univariate:**
- Each feature analyzed independently
- Miss joint outliers

**Example:**
```
Feature 1: [10, 20, 30, 40]  ← All normal
Feature 2: [5,  10, 15, 20]  ← All normal
Combined: (10,5), (20,10), (30,15), (40,20)  ← Last point is outlier!
```

**Multivariate:**
- Considers feature correlations
- Uses Mahalanobis distance
- Detects joint outliers

### 6.4 Interview Questions

**Technical (Data Scientist):**

**Q:** You detect 30% of your data as outliers. What's wrong?

**A:** Possible issues:
1. **Wrong threshold:** Too aggressive (use domain expertise to set)
2. **Wrong distribution assumption:** Data may not be normal (use IQR instead)
3. **Multiple populations:** Data may have legitimate subgroups (cluster first)
4. **Process change:** Normal range may have shifted (check temporal patterns)

**Q:** How do you choose between IQR and Z-score?**

**Decision Tree:**
```
Is data normally distributed?
├─ Yes → Check for outliers first
│   ├─ Few outliers → Z-score okay
│   └─ Many outliers → Use Modified Z-score
└─ No (skewed/unknown) → Use IQR
```

**Practical (Financial Services):**

**Q:** We're detecting fraudulent transactions. Which method is best?

**A:** **Neither alone!** For fraud detection:
1. **High recall needed:** Can't miss fraud (False Negative is costly)
2. **Use ensemble:** Combine multiple methods
3. **Supervised if possible:** Train classifier with labeled fraud examples
4. **Consider context:** Transaction amount alone insufficient
   - Amount relative to user's history
   - Time of day, location, merchant type
   - Sequence of transactions
5. **Use Isolation Forest or LOF** for complex patterns

---

## Summary

**Key Takeaways:**
1. Anomaly detection identifies rare, significantly different observations
2. IQR method is robust and distribution-free
3. Z-score assumes normality and is sensitive to outliers
4. Modified Z-score is more robust variant
5. Multivariate methods needed for high-dimensional data

**Method Selection Guide:**
- **IQR:** Default choice, works for any distribution
- **Z-Score:** Only if data is truly normal and well-behaved
- **Modified Z:** Compromise between IQR and Z-score
- **Mahalanobis:** Multivariate data with correlations

**Practical Considerations:**
- Always visualize data first
- Try multiple methods and compare
- Use domain knowledge to validate
- Check for temporal patterns
- Consider contextual information
- Handle masking effect in iterative detection

**Common Pitfalls:**
- Assuming normality without checking
- Ignoring domain context
- Using fixed thresholds across all datasets
- Treating all outliers as errors (some may be valuable insights)
- Not checking for multicollinearity in multivariate detection
