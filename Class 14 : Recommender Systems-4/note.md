# Class 14-16: Matrix Factorization Implementation & Optimization
> See Class 13 for comprehensive Matrix Factorization theory.  
> This note covers advanced implementation details.

## Additional Implementation Details

### Alternative Factorization: NMF (Non-Negative Matrix Factorization)

```python
from sklearn.decomposition import NMF

# Ensure ratings are non-negative
ratings_matrix = np.array([[5,3,0,1],[4,0,0,1],[1,1,5,4],[1,0,4,5]])

nmf = NMF(n_components=2, init='random', random_state=42)
W = nmf.fit_transform(ratings_matrix)  # User factors
H = nmf.components_  # Item factors

# Reconstruct
predicted = W @ H
print("NMF Reconstruction:\n", predicted)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_factors': [10, 20, 50],
    'n_epochs': [20, 30],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}

# Use Surprise's GridSearchCV
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(dataset)

print(f"Best RMSE: {gs.best_score['rmse']}")
print(f"Best params: {gs.best_params['rmse']}")
```

**Note:** For complete theory, algorithms, and comprehensive examples, refer to **Class 13** which covers:
- Matrix Factorization fundamentals
- SVD and ALS algorithms  
- SGD optimization
- Hybrid recommender systems
- All exam preparation materials
