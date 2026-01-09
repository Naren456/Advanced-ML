# Class 12: Recommender Systems - Collaborative Filtering

> **Core Principle:** "People who agreed in the past tend to agree in the future"

---

## Table of Contents
1. [Collaborative Filtering Overview](#1-collaborative-filtering-overview)
2. [User-User Collaborative Filtering](#2-user-user-collaborative-filtering)
3. [Item-Item Collaborative Filtering](#3-item-item-collaborative-filtering)
4. [Market Basket Analysis](#4-market-basket-analysis)
5. [Implementation](#5-implementation)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Collaborative Filtering Overview

### 1.1 Core Concept

**Idea:** Leverage collective intelligence of user community

**Example:**
```
User A liked: Movie1, Movie2, Movie3
User B liked: Movie1, Movie2, Movie4
User C (new) liked: Movie1, Movie2

Recommendation for C: Movie3 (from A) or Movie4 (from B)
```

### 1.2 User-Item Matrix

**Representation:**
```
           Movie1  Movie2  Movie3  Movie4  Movie5
UserA         5       4       ?       3       ?
UserB         4       5       2       ?       1
UserC         ?       4       5       2       ?
UserD         3       ?       4       5       ?
```

**Characteristics:**
- **Sparse:** Most entries are missing (users rate few items)
- **High-dimensional:** Millions of users × items
- **Goal:** Predict missing values

---

## 2. User-User Collaborative Filtering

### 2.1 Algorithm Steps

**Step 1:** Find similar users to target user
**Step 2:** Aggregate ratings from similar users
**Step 3:** Recommend highest-rated items

### 2.2 Similarity Calculation

**Pearson Correlation:**
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}}(r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}}(r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}}(r_{vi} - \bar{r}_v)^2}}$$

where $I_{uv}$ = items rated by both users

**Cosine Similarity:**
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2} \sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}$$

### 2.3 Prediction Formula

**Weighted Average:**
$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u,v) \times (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u,v)|}$$

where $N(u)$ = set of similar users who rated item $i$

---

## 3. Item-Item Collaborative Filtering

### 3.1 Why Item-Item?

**Advantages over User-User:**
1. **Stability:** Items don't change, users do
2. **Scalability:** Fewer items than users (typically)
3. **Explainability:** "You liked X, so you'll like Y"
4. **Precomputation:** Can compute similarity offline

### 3.2 Item Similarity

**Adjusted Cosine:**
$$\text{sim}(i, j) = \frac{\sum_{u \in U_{ij}}(r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}}(r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_{ij}}(r_{uj} - \bar{r}_u)^2}}$$

**Why adjust for user mean?** Users have different rating scales

### 3.3 Prediction

$$\hat{r}_{ui} = \frac{\sum_{j \in N(i)} \text{sim}(i,j) \times r_{uj}}{\sum_{j \in N(i)} |\text{sim}(i,j)|}$$

where $N(i)$ = similar items rated by user $u$

---

## 4. Market Basket Analysis

### 4.1 Association Rules

**Goal:** Find relationships between items purchased together

**Example:** {Bread, Butter} → {Milk}

### 4.2 Key Metrics

**Support:**
$$\text{Support}(X) = \frac{\text{Transactions containing } X}{\text{Total transactions}}$$

**Confidence:**
$$\text{Confidence}(X \rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$

**Lift:**
$$\text{Lift}(X \rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) \times \text{Support}(Y)}$$

**Interpretation:**
- Lift > 1: Positive correlation (buying X increases chance of buying Y)
- Lift = 1: Independent
- Lift < 1: Negative correlation

### 4.3 Apriori Algorithm

**Principle:** If itemset is frequent, all subsets are frequent

**Algorithm:**
```
1. Find frequent 1-itemsets (Support > min_support)
2. Generate candidate 2-itemsets from frequent 1-itemsets
3. Prune candidates whose subsets aren't frequent
4. Check support of remaining candidates
5. Repeat for k-itemsets until no more frequent sets
```

---

## 5. Implementation

### 5.1 User-User Collaborative Filtering

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# User-Item rating matrix
ratings = np.array([
    [5, 4, 0, 0, 1],  # User 0
    [4, 0, 0, 2, 1],  # User 1
    [0, 3, 5, 4, 0],  # User 2
    [0, 0, 4, 5, 3]   # User 3
])

# Compute user-user similarity
# Replace 0s with NaN for similarity calculation
ratings_for_sim = ratings.copy().astype(float)
ratings_for_sim[ratings_for_sim == 0] = np.nan

# Calculate mean ratings per user
user_means = np.nanmean(ratings_for_sim, axis=1)

# Center ratings
ratings_centered = ratings_for_sim - user_means[:, np.newaxis]
ratings_centered = np.nan_to_num(ratings_centered)  # Replace NaN with 0

# Compute similarity
user_sim = cosine_similarity(ratings_centered)
np.fill_diagonal(user_sim, 0)  # Don't consider self-similarity

print("User Similarity Matrix:")
print(user_sim)

def predict_rating(user_id, item_id, ratings, user_sim, k=2):
    """
    Predict rating for user-item pair
    """
    # Find k most similar users who rated this item
    rated_mask = ratings[:, item_id] > 0
    similar_users = user_sim[user_id].copy()
    similar_users[~rated_mask] = -1  # Ignore users who didn't rate the item
    
    top_k_users = np.argsort(similar_users)[-k:]
    
    # Compute weighted average
    numerator = sum(user_sim[user_id, u] * ratings[u, item_id] 
                   for u in top_k_users)
    denominator = sum(abs(user_sim[user_id, u]) for u in top_k_users)
    
    if denominator == 0:
        return user_means[user_id]
    
    return numerator / denominator

# Predict rating for User 0, Item 2
pred = predict_rating(0, 2, ratings, user_sim)
print(f"\nPredicted rating for User 0, Item 2: {pred:.2f}")
```

### 5.2 Item-Item Collaborative Filtering

```python
# Compute item-item similarity
# Transpose to get items as rows
ratings_items = ratings.T
ratings_items_for_sim = ratings_items.copy().astype(float)
ratings_items_for_sim[ratings_items_for_sim == 0] = np.nan

# Center by user mean (across columns)
item_centered = ratings_items_for_sim - np.nanmean(ratings_items_for_sim, axis=0)
item_centered = np.nan_to_num(item_centered)

# Compute similarity
item_sim = cosine_similarity(item_centered)
np.fill_diagonal(item_sim, 0)

print("Item Similarity Matrix:")
print(item_sim)

def recommend_items(user_id, ratings, item_sim, top_n=3):
    """
    Recommend top N items for user
    """
    # Get user's ratings
    user_ratings = ratings[user_id]
    
    # Predict scores for unrated items
    unrated_items = np.where(user_ratings == 0)[0]
    predictions = []
    
    for item in unrated_items:
        # Find similar items that user has rated
        rated_items = np.where(user_ratings > 0)[0]
        
        # Weighted sum
        numerator = sum(item_sim[item, rated] * user_ratings[rated] 
                       for rated in rated_items)
        denominator = sum(abs(item_sim[item, rated]) for rated in rated_items)
        
        if denominator > 0:
            pred_score = numerator / denominator
            predictions.append((item, pred_score))
    
    # Return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# Get recommendations for User 0
recs = recommend_items(0, ratings, item_sim)
print(f"\nTop recommendations for User 0:")
for item, score in recs:
    print(f"  Item {item}: {score:.2f}")
```

### 5.3 Market Basket Analysis with Apriori

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Transaction data (basket format)
data = pd.DataFrame([
    [1, 1, 0, 1, 0],  # Transaction 1: Items 0,1,3
    [0, 1, 1, 1, 0],  # Transaction 2: Items 1,2,3
    [1, 1, 0, 0, 1],  # Transaction 3: Items 0,1,4
    [1, 0, 1, 0, 1],  # Transaction 4: Items 0,2,4
    [0, 1, 1, 1, 1],  # Transaction 5: Items 1,2,3,4
], columns=['Bread', 'Milk', 'Butter', 'Beer', 'Diapers'])

# Find frequent itemsets
frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Interpret
print("\nStrong Rules (Lift > 1.2):")
strong_rules = rules[rules['lift'] > 1.2]
for _, rule in strong_rules.iterrows():
    print(f"  {set(rule['antecedents'])} → {set(rule['consequents'])}")
    print(f"    Support: {rule['support']:.2f}, Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}")
```

---

## 6. Exam Preparation

### 6.1 User-User vs Item-Item

| Aspect | User-User | Item-Item |
|--------|-----------|-----------|
| **Similarity** | Between users | Between items |
| **Scalability** | $O(U^2)$ | $O(I^2)$ |
| **Stability** | Users change | Items stable |
| **For large systems** | Poor (millions of users) | Better (fewer items) |
| **Precomputation** | Difficult (users change) | Easy (items stable) |
| **Explainability** | "Users like you..." | "You liked X, try Y" |

**When to use:**
- **Few users, many items:** User-User
- **Many users, few items:** Item-Item (e.g., Amazon, Netflix)

### 6.2 Common Exam Questions

**Q1: Calculate similarity between users U1=[5,3,0,1] and U2=[4,0,4,1].**

**Solution (Cosine Similarity on common items):**

Common items: indices 0 and 3
- U1_common = [5, 1]
- U2_common = [4, 1]

$$\text{sim} = \frac{5 \times 4 + 1 \times 1}{\sqrt{5^2+1^2} \times \sqrt{4^2+1^2}} = \frac{21}{\sqrt{26} \times \sqrt{17}} = \frac{21}{21.02} = 0.999$$

**Q2: Given association rule {Milk} → {Bread} with Support=0.3, Confidence=0.6. If Support(Bread)=0.4, calculate Lift.**

**Solution:**
$$\text{Lift} = \frac{\text{Confidence}(Milk \to Bread)}{\text{Support}(Bread)} = \frac{0.6}{0.4} = 1.5$$

Interpretation: Buying milk increases likelihood of buying bread by 50%

**Q3: Explain the cold-start problem for collaborative filtering.**

**Answer:**
1. **New User:** No ratings → can't find similar users → can't recommend
   - Solution: Ask for initial preferences, use popularity, demographic info
2. **New Item:** No ratings → won't be recommended
   - Solution: Use content-based initially, promote to active users

### 6.3 Interview Questions

**Q: Amazon has millions of users and products. Which collaborative filtering approach?**

**A: Item-Item Collaborative Filtering**  
Reasons:
1. Fewer products than users (Items < Users)
2. Products don't change rating patterns (stable)
3. Can precompute item-item similarity overnight
4. Better explainability: "Customers who bought X also bought Y"
5. Scales well with user growth

**Q: How to handle rating scale differences (some users rate 3-5, others 1-5)?**

**A: Normalize ratings**
1. **Mean-centering:** Subtract user's mean rating
$$r'_{ui} = r_{ui} - \bar{r}_u$$

2. **Z-score normalization:**
$$r'_{ui} = \frac{r_{ui} - \bar{r}_u}{\sigma_u}$$

This ensures similarity isn't affected by individual rating distributions.

---

## Summary

**Key Takeaways:**
- Collaborative filtering uses user collective behavior
- User-User: Find similar users, aggregate their preferences
- Item-Item: Find similar items, recommend based on user's history
- Market Basket: Association rules for cross-selling
- Apriori algorithm mines frequent itemsets efficiently

**Advantages:**
- No need for item metadata
- Discovers unexpected patterns
- Quality improves with more data
- Works across domains

**Limitations:**
- Cold-start problem (new users/items)
- Sparsity (most cells empty)
- Scalability challenges
- Cannot recommend new items until rated
- Popularity bias (rich get richer)

**Best Practices:**
- Use Item-Item for large-scale systems
- Normalize ratings to handle scale differences
- Combine with content-based (hybrid)
- Handle sparsity with matrix factorization
- Use implicit feedback when explicit ratings unavailable
