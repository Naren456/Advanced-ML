# Class 07: Anomaly Detection (Part 1)

## Overview
**Anomaly Detection** (or Outlier Detection) is the identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

## 1. IQR (Interquartile Range) Method
*   **Concept**: Uses the statistical range between the 1st and 3rd quartiles to define "normal" data boundaries.
*   **Steps**:
    1.  Calculate **Q1** (25th percentile) and **Q3** (75th percentile).
    2.  Calculate **IQR** = $Q3 - Q1$.
    3.  Define Bounds:
        *   **Lower Bound**: $Q1 - 1.5 \times IQR$ (Note: The notebook code snippet uses Q3 - 1.5*IQR which might be a typo, standard is Q1).
        *   **Upper Bound**: $Q3 + 1.5 \times IQR$ (Note: The notebook code snippet uses Q1 + 1.5*IQR which might be a typo, standard is Q3).
    4.  Any data point outside these bounds is considered an outlier.
*   **Robustness**: IQR is robust to extreme outliers because it depends on quartiles rather than mean/variance.

## 2. Z-Score Method
*   **Concept**: Measures how many standard deviations a data point is away from the mean.
*   **Formula**:
    $$z = \frac{x - \mu}{\sigma}$$
    Where $x$ is the data point, $\mu$ is the mean, and $\sigma$ is the standard deviation.
*   **Threshold**: Commonly, if $|z| > 2$ or $|z| > 3$, the point is flagged as an outlier.
*   **Assume**: Assumes the data follows a Gaussian (Normal) distribution.

## Code Implementation
**Notebook**: `colab.ipynb`
*   **Libraries**: `pandas`
*   **Implementation**:
    *   `IOR_Detection(df)`: Iterates through columns to find outliers using the IQR rule.
    *   `z_score_outlier(df, threshold=2)`: Calculates Z-scores for each value and identifies indices where the Z-score exceeds the threshold.
    *   Example using a simple DataFrame with 'age' and 'salary' columns containing obvious outliers (e.g., age 80, salary 339 or 200).
