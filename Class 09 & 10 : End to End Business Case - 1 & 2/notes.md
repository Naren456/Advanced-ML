# Class 09 & 10: End-to-End Business Case - Customer Segmentation

> **Core Principle:** "Integrating multiple techniques to solve real business problems"

---

## Table of Contents
1. [Business Problem Definition](#1-business-problem-definition)
2. [Data Preprocessing Pipeline](#2-data-preprocessing-pipeline)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Dimensionality Reduction](#4-dimensionality-reduction)
5. [Anomaly Detection](#5-anomaly-detection)
6. [Clustering](#6-clustering)
7. [Business Insights](#7-business-insights)
8. [Exam Preparation](#8-exam-preparation)

---

## 1. Business Problem Definition

### 1.1 Case Study: Mall Customer Segmentation

**Dataset:** `Mall_Customers.csv`  
**Objective:** Group customers into meaningful segments for targeted marketing

**Business Questions:**
1. Who are our most valuable customers?
2. Which customers should we target with which campaigns?
3. Are there underserved customer segments?
4. What characteristics define each segment?

### 1.2 Available Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| CustomerID | Categorical | Unique identifier | 1-200 |
| Gender | Categorical | Male/Female | Binary |
| Age | Numerical | Customer age | 18-70 |
| Annual Income | Numerical | Income in thousands | 15-137 |
| Spending Score | Numerical | Score assigned (1-100) | 1-99 |

### 1.3 Success Metrics

**Quantitative:**
- Silhouette score > 0.5
- Well-separated clusters (Davies-Bouldin < 1.0)
- Explained variance > 80%

**Qualitative:**
- Business interpretability
- Actionable insights
- Distinct marketing strategies per segment

---

## 2. Data Preprocessing Pipeline

### 2.1 Data Loading and Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Mall_Customers.csv')

# Initial inspection
print("Dataset Shape:", df.shape)
print("\nFirst Few Rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
```

### 2.2 Data Cleaning

**Step 1: Handle Missing Values**
```python
# Check for missing values
missing = df.isnull().sum()
print(f"Missing values:\n{missing}")

# For this dataset: typically no missing values
# If present, strategies:
# - Mode for categorical (Gender)
# - Median for numerical (Age, Income, Score)
```

**Step 2: Remove Duplicates**
```python
# Check duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Remove if any
df = df.drop_duplicates()
```

**Step 3: Drop Non-Informative Features**
```python
# CustomerID is just an identifier, doesn't add predictive value
df = df.drop('CustomerID', axis=1)
```

### 2.3 Feature Engineering

**Encode Categorical Variables:**
```python
# Gender: Male=1, Female=0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
```

**Create Derived Features (Optional):**
```python
# Income-to-Spending Ratio
df['Income_Spending_Ratio'] = df['Annual Income (k$)'] / (df['Spending Score (1-100)'] + 1)

# Age Groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                         labels=['Young', 'Middle', 'Senior', 'Elderly'])
```

### 2.4 Feature Scaling

**Why:** K-Means is distance-based → features must be on same scale

```python
from sklearn.preprocessing import StandardScaler

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify scaling
print("Mean after scaling:", X_scaled.mean(axis=0))  # Should be ~0
print("Std after scaling:", X_scaled.std(axis=0))    # Should be ~1
```

---

## 3. Exploratory Data Analysis

### 3.1 Univariate Analysis

```python
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gender distribution
df['Gender'].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Gender Distribution')
axes[0,0].set_xticklabels(['Female', 'Male'], rotation=0)

# Age distribution
axes[0,1].hist(df['Age'], bins=20, edgecolor='black')
axes[0,1].set_title('Age Distribution')
axes[0,1].set_xlabel('Age')

# Income distribution
axes[1,0].hist(df['Annual Income (k$)'], bins=20, edgecolor='black')
axes[1,0].set_title('Annual Income Distribution')
axes[1,0].set_xlabel('Income (k$)')

# Spending Score distribution
axes[1,1].hist(df['Spending Score (1-100)'], bins=20, edgecolor='black')
axes[1,1].set_title('Spending Score Distribution')
axes[1,1].set_xlabel('Score')

plt.tight_layout()
plt.show()
```

### 3.2 Bivariate Analysis

```python
# Most important: Income vs Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
           c=df['Gender'], cmap='coolwarm', alpha=0.6, s=100)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Income vs Spending Score (colored by Gender)')
plt.colorbar(label='Gender (0=F, 1=M)')
plt.grid(True, alpha=0.3)
plt.show()
```

**Key Observation:** Visual clusters appear!
```
High income, High spending: Target customers
High income, Low spending: Potential to convert
Low income, High spending: Credit risk?
Low income, Low spending: Budget shoppers
```

### 3.3 Correlation Analysis

```python
# Correlation matrix
plt.figure(figsize=(8, 6))
correlation = df[features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
           square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.show()

# Typical findings:
# - Income and Spending Score: Low correlation (~0.0)
# - Age and Income: Weak positive (~0.2)
# - Age and Spending: Weak negative (~-0.3)
```

---

## 4. Dimensionality Reduction

### 4.1 PCA for Visualization

**Objective:** Reduce 3D+ data to 2D for visualization

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA: 3D → 2D Projection')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.2 t-SNE for Non-Linear Patterns

```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Projection (Preserves Local Structure)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Anomaly Detection

### 5.1 Purpose

**Why detect anomalies before clustering?**
1. Outliers distort cluster centroids
2. Unusual customers may need special handling
3. Data quality check

### 5.2 Multi-Method Approach

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Method 1: IQR on key feature (Spending Score)
Q1 = df['Spending Score (1-100)'].quantile(0.25)
Q3 = df['Spending Score (1-100)'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = (df['Spending Score (1-100)'] < (Q1 - 1.5*IQR)) | \
               (df['Spending Score (1-100)'] > (Q3 + 1.5*IQR))

# Method 2: Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers_if = iso_forest.fit_predict(X_scaled) == -1

# Method 3: LOF
lof = LocalOutlierFactor(contamination=0.05)
outliers_lof = lof.fit_predict(X_scaled) == -1

# Combine (outlier if 2+ methods agree)
outliers_combined = outliers_iqr.values + outliers_if + outliers_lof >= 2

print(f"Outliers detected: {outliers_combined.sum()}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(df.loc[~outliers_combined, 'Annual Income (k$)'],
           df.loc[~outliers_combined, 'Spending Score (1-100)'],
          c='blue', label='Normal', alpha=0.6)
plt.scatter(df.loc[outliers_combined, 'Annual Income (k$)'],
           df.loc[outliers_combined, 'Spending Score (1-100)'],
           c='red', marker='x', s=200, label='Outlier', linewidths=3)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Anomaly Detection Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.3 Decision on Outliers

**Options:**
1. **Remove:** If data quality issue
2. **Separate segment:** "VIP" or "Unusual" customers
3. **Keep:** If legitimate rare customers

**For this case:** Usually keep (represent real customer diversity)

---

## 6. Clustering

### 6.1 Determine Optimal K

**Method 1: Elbow Method**

```python
from sklearn.cluster import KMeans

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=10)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)
plt.axvline(x=5, color='r', linestyle='--', label='Optimal K=5')
plt.legend()
plt.show()
```

**Method 2: Silhouette Score**

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=10)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True, alpha=0.3)
plt.show()
```

**Finding:** K=5 shows good balance (elbow point + high silhouette)

### 6.2 Final Clustering

```python
# Fit final model with K=5
kmeans_final = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Cluster centers (in original scale)
centers_scaled = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

print("Cluster Centers (Original Scale):")
print(pd.DataFrame(centers_original, columns=features))
```

### 6.3 Visualization

```python
# 2D visualization (Income vs Spending)
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               c=colors[cluster], label=f'Cluster {cluster}',
               alpha=0.6, s=100)

# Plot centroids
centers_orig = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centers_orig[:, 1], centers_orig[:, 2],  # Income, Spending
           c='black', marker='X', s=300, linewidths=3,
           edgecolors='yellow', label='Centroids')

plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Customer Segments (K-Means Clustering)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 6.4 Alternative: DBSCAN

```python
from sklearn.cluster import DBSCAN

# Try DBSCAN for comparison
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

n_clusters = len(set(df['Cluster_DBSCAN'])) - (1 if -1 in df['Cluster_DBSCAN'] else 0)
n_noise = list(df['Cluster_DBSCAN']).count(-1)

print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
           c=df['Cluster_DBSCAN'], cmap='viridis', alpha=0.6)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
```

---

## 7. Business Insights

### 7.1 Cluster Profiling

```python
# Statistical summary by cluster
cluster_summary = df.groupby('Cluster')[features].mean()
print("Cluster Profiles:")
print(cluster_summary)

# Add size information
cluster_summary['Count'] = df.groupby('Cluster').size()
cluster_summary['Percentage'] = (cluster_summary['Count'] / len(df) * 100).round(1)
print("\nCluster Sizes:")
print(cluster_summary[['Count', 'Percentage']])
```

### 7.2 Segment Interpretation

**Typical 5-Cluster Solution:**

| Cluster | Profile | Income | Spending | Age | Strategy |
|---------|---------|--------|----------|-----|----------|
| **0** | Budget Conscious | Low | Low | Mixed | Value deals, discounts |
| **1** | High Value | High | High | Young | Premium products, loyalty program |
| **2** | Careful Spenders | High | Low | Senior | Quality focus, trust-building |
| **3** | Careless Young | Low | High | Young | Payment plans, credit offers |
| **4** | Average | Medium | Medium | Middle | Standard campaigns |

### 7.3 Actionable Recommendations

```python
# Generate segment-specific insights
segments = {
    0: {
        'name': 'Budget Shoppers',
        'description': 'Low income, low spending',
        'strategy': [
            'Discount campaigns',
            'Bundle deals',
            'Loyalty points',
            'Email with weekly specials'
        ]
    },
    1: {
        'name': 'VIP Customers',
        'description': 'High income, high spending',
        'strategy': [
            'Premium product launches',
            'Exclusive events',
            'Personal shopper service',
            'VIP lounge access'
        ]
    },
    # ... (continue for all clusters)
}

# Print recommendations
for cluster_id, segment in segments.items():
    cluster_size = len(df[df['Cluster'] == cluster_id])
    print(f"\n{'='*60}")
    print(f"Cluster {cluster_id}: {segment['name']}")
    print(f"Size: {cluster_size} customers ({cluster_size/len(df)*100:.1f}%)")
    print(f"Profile: {segment['description']}")
    print("Recommended Strategies:")
    for strategy in segment['strategy']:
        print(f"  - {strategy}")
```

### 7.4 Visualization Dashboard

```python
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Cluster distribution
ax1 = fig.add_subplot(gs[0, :])
cluster_counts = df['Cluster'].value_counts().sort_index()
ax1.bar(cluster_counts.index, cluster_counts.values, color=colors)
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Number of Customers')
ax1.set_title('Cluster Size Distribution')

# 2. Income vs Spending (main viz)
ax2 = fig.add_subplot(gs[1, :2])
for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    ax2.scatter(cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)
ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score (1-100)')
ax2.set_title('Income vs Spending by Cluster')
ax2.legend()

# 3. Age distribution by cluster
ax3 = fig.add_subplot(gs[1, 2])
df.boxplot(column=['Age'], by='Cluster', ax=ax3)
ax3.set_title('Age Distribution by Cluster')
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Age')

# 4. Gender distribution by cluster
ax4 = fig.add_subplot(gs[2, 0])
gender_cluster = pd.crosstab(df['Cluster'], df['Gender'])
gender_cluster.plot(kind='bar', stacked=True, ax=ax4, color=['pink', 'lightblue'])
ax4.set_title('Gender Distribution by Cluster')
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Count')
ax4.legend(['Female', 'Male'])

# 5. 3D visualization (if 3+ features)
from mpl_toolkits.mplot3d import Axes3d
ax5 = fig.add_subplot(gs[2, 1:], projection='3d')
for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]
    ax5.scatter(cluster_data['Age'],
               cluster_data['Annual Income (k$)'],
               cluster_data['Spending Score (1-100)'],
               c=colors[cluster], label=f'Cluster {cluster}', s=50)
ax5.set_xlabel('Age')
ax5.set_ylabel('Income')
ax5.set_zlabel('Spending Score')
ax5.set_title('3D Cluster Visualization')

plt.show()
```

---

## 8. Exam Preparation

### 8.1 End-to-End Workflow

```
1. Business Understanding
   ├─ Define problem
   ├─ Identify stakeholders
   └─ Set success metrics

2. Data Collection & Cleaning
   ├─ Load data
   ├─ Handle missing values
   ├─ Remove duplicates
   └─ Feature engineering

3. Exploratory Data Analysis
   ├─ Univariate analysis
   ├─ Bivariate analysis
   └─ Correlation analysis

4. Preprocessing
   ├─ Encode categoricals
   ├─ Scale numericals
   └─ Handle outliers

5. Optional: Dimensionality Reduction
   ├─ PCA for visualization
   └─ t-SNE for complex patterns

6. Anomaly Detection
   ├─ Multiple methods
   ├─ Consensus approach
   └─ Handle outliers

7. Clustering
   ├─ Determine optimal K
   ├─ Apply algorithm (K-Means/DBSCAN/GMM)
   └─ Validate results

8. Interpretation & Action
   ├─ Profile each cluster
   ├─ Business insights
   └─ Actionable recommendations
```

### 8.2 Common Exam Questions

**Q1: Why scale features before clustering?**

**Answer:** Distance-based algorithms (K-Means, DBSCAN) are sensitive to feature scales. Without scaling:
- Features with larger ranges dominate distance calculations
- Example: Income (0-200k) will overwhelm Age (18-70)
- StandardScaler ensures equal contribution: $z = \frac{x - \mu}{\sigma}$

**Q2: How to choose between K-Means and DBSCAN?**

| Criterion | Choose K-Means | Choose DBSCAN |
|-----------|---------------|----------------|
| Cluster shape | Spherical | Arbitrary |
| Number of clusters | Known | Unknown |
| Outliers | Few | Many |
| Cluster density | Uniform | Varying |

**Q3: What if elbow method is ambiguous?**

**Solution:** Use multiple criteria:
1. **Silhouette Score:** Maximize
2. **Davies-Bouldin Index:** Minimize
3. **Gap Statistic:** Compare to random data
4. **Business Constraints:** "We can only support 3-5 segments"
5. **Domain Expertise:** Does 5 clusters make sense?

### 8.3 Interview Questions

**Q:** You segment customers and find one cluster is 80% of data. Is this useful?

**A:** **Probably not.** This suggests:
1. **K too small:** Increase number of clusters
2. **Poor feature selection:** Need more discriminative features
3. **Imbalanced data:** May need stratified sampling or different algorithm
4. **Actually homogeneous:** Maybe customers ARE similar (rare)

**Action:** 
- Try higher K
- Add more features
- Use hierarchical clustering to subdivide large cluster

**Q:** How to present clustering results to non-technical stakeholders?

**A:** Focus on business value:
1. **Clear names:** Not "Cluster 0" but "Budget Shoppers"
2. **Simple visualizations:** 2D scatter plots, bar charts
3. **Actionable insights:** "Target Cluster 1 with premium products"
4. **Success metrics:** "Expected 15% revenue increase from Cluster 1"
5. **Examples:** Show 3-5 typical customers per segment
6. **avoid jargon:** No "silhouette scores" or "inertia"

---

## Summary

**Key Takeaways:**
1. Business cases require integrating multiple ML techniques
2. Workflow: Understand → Clean → Explore → Preprocess → Model → Interpret
3. Always validate results with business logic
4. Visualization is crucial for communication
5. Actionable insights matter more than perfect metrics

**Critical Steps:**
- Feature scaling before clustering
- Multiple methods for optimal K
- Validate with domain expertise
- Profile clusters thoroughly
- Generate actionable strategies

**Common Pitfalls:**
- Skipping EDA
- Not scaling features
- Choosing K arbitrarily
- Ignoring business context
- Over-complicating presentation

**Real-World Considerations:**
- Data may be messy
- Stakeholders want simple answers
- Segments must be actionable
- Results should drive decisions
- Model needs periodic retraining