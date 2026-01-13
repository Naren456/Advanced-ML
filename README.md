# Advanced ML - Complete Course Notes

## Overview
Professional, textbook-quality notes for all 22 classes covering Dimensionality Reduction, Clustering, Anomaly Detection, Recommender Systems, and Time Series Analysis.

**All classes have individual, comprehensive, standalone notes.**

---

## Course Structure

### Dimensionality Reduction (Classes 01-03)

| Class | Topic | Key Content |
|-------|-------|-------------|
| **01** | Principal Component Analysis (PCA) | Eigenvalue decomposition, curse of dimensionality, variance maximization |
| **02** | t-SNE | Probabilistic embedding, crowding problem solution, KL divergence |
| **03** | t-SNE Advanced & UMAP | Topological methods, comparative analysis, parameter tuning |

### Clustering (Classes 04-06)

| Class | Topic | Key Content |
|-------|-------|-------------|
| **04** | K-Means & K-Means++ | Algorithm convergence, elbow method, silhouette analysis |
| **05** | Gaussian Mixture Models (GMM) | EM algorithm, soft clustering, covariance types (spherical/full) |
| **06** | DBSCAN | Density-based clustering, core/border/noise points, parameter selection |

### Anomaly Detection (Classes 07-08)

| Class | Topic | Key Content |
|-------|-------|-------------|
| **07** | Statistical Methods | IQR method, Z-score, modified Z-score, robust variants |
| **08** | Machine Learning Methods | Isolation Forest, LOF, One-Class SVM, comparative analysis |

### Business Case (Classes 09-10)

| Class | Topic | Key Content |
|-------|-------|-------------|
| **09-10** | End-to-End Customer Segmentation | Complete ML pipeline, EDA, preprocessing, clustering, insights |

### Recommender Systems (Classes 11-16)

| Class | Topic | Key Content |
|-------|-------|-------------|
| **11** | Content-Based Filtering | TF-IDF, cosine similarity, feature engineering |
| **12** | Collaborative Filtering | User-user, item-item, market basket analysis, Apriori |
| **13** | Matrix Factorization | SGD optimization, latent factors, comprehensive theory |
| **14** | Advanced MF - Implementation | NMF, hyperparameter tuning, grid search |
| **15** | Advanced MF - Deep Learning | Neural collaborative filtering, context-aware systems |
| **16** | Production Deployment | Model evaluation, deployment checklist, API design |

### Time Series Analysis (Classes 17-22)

| Class | Topic | Key Content |
|-------|-------|-------------|
| **17** | Introduction & Decomposition | Components (T,S,C,R), additive vs multiplicative, data prep |
| **18** | Stationarity & Baselines | ADF/KPSS tests, differencing, naive/drift methods |
| **19** | Smoothing Techniques | Moving average, SES, Holt's method, Holt-Winters |
| **20** | Advanced Smoothing | Dampened trends, Box-Cox transformation, model selection |
| **21** | ACF, PACF & ARIMA | Pattern recognition, model identification, order selection |
| **22** | SARIMA & Production | Seasonal ARIMA, diagnostics, deployment framework |

---


## How to Use These Notes

### For Exam Preparation
1. Read "Exam Preparation" section in each note
2. Work through numerical examples
3. Practice derivations
4. Review formula summary tables
5. Test yourself with practice questions

### For Interviews
1. Focus on "Interview Questions" sections
2. Understand intuition behind algorithms
3. Practice explaining concepts simply
4. Review comparison tables
5. Know when to use each method

### For Implementation
1. Study code examples in each class
2. Run and modify provided implementations
3. Follow production deployment patterns
4. Understand parameter tuning
5. Practice on real datasets

### For Deep Understanding
1. Work through mathematical derivations
2. Implement algorithms from scratch
3. Visualize results
4. Compare multiple methods
5. Read research papers referenced

---

## Quick Reference Guide

### Algorithm Selection

**Dimensionality Reduction:**
- Need interpretability? → **PCA**
- Visualization of clusters? → **t-SNE**
- Balance local/global structure? → **UMAP**

**Clustering:**
- Known number of clusters, spherical? → **K-Means**
- Soft assignments needed? → **GMM**
- Arbitrary shapes, unknown K? → **DBSCAN**

**Anomaly Detection:**
- Simple, robust, no assumptions? → **IQR**
- Gaussian data? → **Z-score**
- Large scale, fast? → **Isolation Forest**
- Varying density? → **LOF**

**Recommender Systems:**
- Rich item metadata? → **Content-Based**
- User behavior patterns? → **Collaborative Filtering**
- Large scale, sparse? → **Matrix Factorization**
- Best accuracy? → **Hybrid (combine methods)**

**Time Series:**
- No trend/seasonality? → **SES**
- Trend only? → **Holt's Method**
- Trend + seasonality? → **Holt-Winters**
- Complex patterns? → **ARIMA/SARIMA**

---

## File Structure

```
Advanced ML/
├── README.md (this file)
│
├── Class 01: PCA/
│   └── note.md
│
├── Class 02: t-SNE/
│   └── note.md
│
├── Class 03: UMAP/
│   └── note.md
│
├── Class 04: K-Means/
│   └── note.md
│
├── Class 05: GMM/
│   └── note.md
│
├── Class 06: DBSCAN/
│   └── note.md
│
├── Class 07: Anomaly Detection 1/
│   └── note.md
│
├── Class 08: Anomaly Detection 2/
│   └── note.md
│
├── Class 09 & 10: Business Case/
│   └── notes.md
│
├── Class 11: Recommender Systems-1/
│   └── note.md
│
├── Class 12: Recommender Systems-2/
│   └── note.md
│
├── Class 13: Recommender Systems-3/
│   └── note.md
│
├── Class 14: Recommender Systems-4/
│   └── note.md
│
├── Class 15: Recommender Systems-5/
│   └── note.md
│
├── Class 16: Recommender Systems-6/
│   └── note.md
│
├── Class 17: Time Series Intro/
│   └── note.md
│
├── Class 18: Time Series Analysis-1/
│   └── note.md
│
├── Class 19: Time Series Analysis-2/
│   └── note.md
│
├── Class 20: Time Series Analysis-3/
│   └── notes.md
│
├── Class 21: Time Series Analysis-4/
│   └── notes.md
│
└── Class 22: Time Series Analysis-5/
    └── notes.md
```

---

## Key Formulas Quick Reference

### PCA
```
Covariance Matrix: Σ = (1/n)X^T X
Eigenvalue Problem: Σv = λv
Explained Variance: λᵢ / Σλⱼ
```

### t-SNE
```
High-D Similarity: pⱼ|ᵢ = exp(-||xᵢ-xⱼ||²/2σᵢ²) / Σ exp(...)
Low-D Similarity: qᵢⱼ = (1+||yᵢ-yⱼ||²)⁻¹ / Σ(1+||yₖ-yₗ||²)⁻¹
Cost: KL(P||Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ/qᵢⱼ)
```

### K-Means
```
Objective: min Σⱼ Σₓ∈Sⱼ ||x - cⱼ||²
Update: cⱼ = (1/|Sⱼ|) Σₓ∈Sⱼ x
Silhouette: s(i) = (b(i)-a(i)) / max{a(i),b(i)}
```

### GMM
```
Mixture: p(x) = Σₖ πₖ N(x|μₖ,Σₖ)
Responsibility: γᵢₖ = πₖN(xᵢ|μₖ,Σₖ) / Σⱼ πⱼN(xᵢ|μⱼ,Σⱼ)
M-Step: μₖ = Σᵢ γᵢₖxᵢ / Σᵢ γᵢₖ
```

### Matrix Factorization
```
Objective: min Σ(rᵢⱼ - pᵢ·qⱼ)² + λ(||P||² + ||Q||²)
SGD Update: pᵢ ← pᵢ + α(eᵢⱼqⱼ - λpᵢ)
```

### ARIMA
```
AR(p): Yₜ = φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ
MA(q): Yₜ = εₜ + θ₁εₜ₋₁ + ... + θ_qεₜ₋q
ARIMA(p,d,q): φ(B)(1-B)ᵈYₜ = θ(B)εₜ
```

---

## Dependencies

All implementations use standard Python libraries:

```python
# Core
numpy
pandas
matplotlib
seaborn
scipy

# Machine Learning
scikit-learn
statsmodels

# Time Series
pmdarima

# Recommenders
surprise (for SVD)

# Dimensionality Reduction
umap-learn
```

Install all:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels pmdarima umap-learn
```

---

## Study Tips

1. **Start with fundamentals** (Classes 1-6) before advanced topics
2. **Implement from scratch** once to understand deeply
3. **Compare methods** on same dataset to see differences
4. **Focus on intuition** before memorizing formulas
5. **Practice explaining** concepts in simple terms
6. **Work through exam questions** repeatedly
7. **Code while reading** - don't just read passively

---

## Additional Resources

Each note is self-contained, but for deeper learning:

- **Research Papers:** Original algorithm papers (linked in notes)
- **Textbooks:** 
  - "Pattern Recognition and Machine Learning" (Bishop)
  - "The Elements of Statistical Learning" (Hastie et al.)
  - "Forecasting: Principles and Practice" (Hyndman & Athanasopoulos)
- **Documentation:** scikit-learn, statsmodels official docs
- **Practice:** Kaggle datasets and competitions

---

## Author Notes

These notes were created as comprehensive exam preparation and practical reference material. They follow professional textbook standards with:

- Mathematical rigor
- Practical implementations
- Real-world examples
- Industry best practices

All code has been tested and verified. Use these notes with confidence for exams, interviews, and production implementations.

**Last Updated:** January 2026

---

## License

Educational use only. Created for Advanced ML coursework.
