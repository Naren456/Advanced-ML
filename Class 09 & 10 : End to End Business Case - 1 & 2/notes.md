# Class 09 & 10: End to End Business Case - Customer Segmentation

## 1. Overview
This case study focuses on customer segmentation using the `Mall_Customers.csv` dataset. The workflow includes Exploratory Data Analysis (EDA), dimensionality reduction, anomaly detection, and clustering to identify distinct customer groups.

## 2. Data Preprocessing
*   **Loading:** Data is loaded using pandas.
*   **Cleaning:**
    *   Checked for duplicates (`df.duplicated().sum()`).
    *   Dropped the `CustomerID` column as it's non-informative.
    *   Mapped `Gender` to numerical values: Male = 1, Female = 0.
*   **Standardization:** Data is scaled using `StandardScaler` for algorithm compatibility.

## 3. Exploratory Data Analysis (EDA)
*   **Univariate Analysis:** `histplot` used for Gender, Age, and Annual Income distributions.
*   **Bivariate Analysis:** `scatterplot` used to visualize relationships:
    *   Age vs. Spending Score
    *   Annual Income vs. Spending Score (hue=Gender)
    *   Age vs. Annual Income
*   **Multivariate Analysis:** Correlation heatmap (`df.corr()`) to identify feature relationships.

## 4. Dimensionality Reduction
Techniques applied to visualize high-dimensional data in 2D:
*   **PCA (Principal Component Analysis):** Reduced scaled data to 2 components. Explained variance ratio is calculated.
*   **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Visualized clusters using `tsne_1` and `tsne_2`.

## 5. Anomaly Detection
Identifying outliers using multiple methods:
*   **IQR (Interquartile Range):** Calculated on `Annual Income` and `Spending Score`. Outliers defined as values outside $[Q1 - 1.5*IQR, Q3 + 1.5*IQR]$.
*   **Z-Score:** Identifies points with z-score > threshold (default 2).
*   **Isolation Forest:** Ensemble method for outlier detection (`contamination=0.05`).
*   **Local Outlier Factor (LOF):** Density-based unsupervised outlier detection (`contamination=0.05`).

## 6. Clustering
*   **K-Means:**
    *   **Elbow Method:** Plotting inertia for k=2 to 10 suggests an optimal **k=5**.
    *   Model fitted with `k=5` and visualized on PCA components.
*   **DBSCAN:**
    *   Density-Based Spatial Clustering of Applications with Noise.
    *   Parameters: `eps=0.5`, `min_samples=5`.
