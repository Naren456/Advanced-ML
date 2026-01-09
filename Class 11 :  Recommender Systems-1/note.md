# Class 11: Recommender Systems - Content-Based Filtering

> **Core Principle:** "Recommend items similar to those a user has liked before"

---

## Table of Contents
1. [Introduction to Recommender Systems](#1-introduction)
2. [Content-Based Filtering Theory](#2-content-based-filtering)
3. [TF-IDF and Cosine Similarity](#3-tfidf-and-cosine-similarity)
4. [Implementation](#4-implementation)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. Introduction

### 1.1 Types of Recommender Systems

**1. Popularity-Based**
- Recommend trending/most-viewed items
- Pros: Simple, works for new users
- Cons: No personalization

**2. Content-Based Filtering**
- Recommend items similar to user's past preferences
- Uses item features/metadata

**3. Collaborative Filtering**
- "Users who liked X also liked Y"
- Uses user-item interaction patterns

**4. Hybrid**
- Combines multiple approaches

### 1.2 Applications

| Domain | Example | Key Feature |
|--------|---------|-------------|
| **E-commerce** | Amazon | Product attributes |
| **Streaming** | Netflix, Spotify | Genre, actors, artists |
| **News** | Google News | Article topics, keywords |
| **Social** | Instagram | Content type, hashtags |

---

## 2. Content-Based Filtering

### 2.1 Core Idea

**Principle:** If user liked Item A, recommend items with similar attributes to A

**Process:**
```
User Profile (Preferences)
        ↓
  Match Against
        ↓
Item Features (Metadata)
        ↓
  Similarity Score
        ↓
  Top N Recommendations
```

### 2.2 Item Representation

**Feature Vector:** Represent each item as vector of attributes

**Example - Movies:**
```
Movie: "The Dark Knight"
Vector: [
  Action: 0.9,
  Drama: 0.3,
  Romance: 0.0,
  Sci-Fi: 0.2,
  Director_Nolan: 1.0,
  Actor_Bale: 1.0
]
```

### 2.3 User Profile Creation

**Method 1: Weighted Average**
$$\text{UserProfile} = \frac{\sum_{i \in \text{liked}} \text{rating}_i \times \text{ItemVector}_i}{\sum_{i \in \text{liked}} \text{rating}_i}$$

**Method 2: Binary (Liked/Not Liked)**
$$\text{UserProfile} = \frac{1}{|\text{liked}|} \sum_{i \in \text{liked}} \text{ItemVector}_i$$

### 2.4 Similarity Measures

**Cosine Similarity:**
$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

**Range:** [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

**Euclidean Distance:**
$$d(A, B) = \sqrt{\sum_{i=1}^n (A_i - B_i)^2}$$

**Pearson Correlation:**
$$r = \frac{\sum(A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum(A_i - \bar{A})^2} \sqrt{\sum(B_i - \bar{B})^2}}$$

---

## 3. TF-IDF and Cosine Similarity

### 3.1 TF-IDF (Term Frequency-Inverse Document Frequency)

**Purpose:** Convert text to numerical vectors

**Term Frequency:**
$$\text{TF}(t, d) = \frac{\text{# times term } t \text{ appears in doc } d}{\text{Total terms in doc } d}$$

**Inverse Document Frequency:**
$$\text{IDF}(t) = \log \frac{\text{Total documents}}{\text{# documents containing term } t}$$

**TF-IDF Score:**
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**Intuition:**
- High TF: Term appears frequently in document (important for this doc)
- High IDF: Term is rare across documents (discriminative)
- Common words (the, is, a) get low IDF scores

### 3.2 Example Calculation

**Corpus:**
- Doc1: "the cat sat on the mat"
- Doc2: "the dog sat on the log"
- Doc3: "cats and dogs are pets"

**For term "cat" in Doc1:**

$$\text{TF}(\text{cat}, \text{Doc1}) = \frac{1}{6} = 0.167$$

$$\text{IDF}(\text{cat}) = \log \frac{3}{2} = 0.176$$

$$\text{TF-IDF}(\text{cat}, \text{Doc1}) = 0.167 \times 0.176 = 0.029$$

**For term "the" in Doc1:**

$$\text{TF}(\text{the}, \text{Doc1}) = \frac{2}{6} = 0.333$$

$$\text{IDF}(\text{the}) = \log \frac{3}{2} = 0.176$$

$$\text{TF-IDF}(\text{the}, \text{Doc1}) = 0.333 \times 0.176 = 0.059$$

---

## 4. Implementation

### 4.1 Content-Based Movie Recommender

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
movies = pd.DataFrame({
    'title': ['The Dark Knight', 'Iron Man', 'The Notebook', 'Titanic', 'Inception'],
    'genres': ['Action|Drama', 'Action|Sci-Fi', 'Romance|Drama', 'Romance|Drama', 'Sci-Fi|Thriller'],
    'description': [
        'Batman fights Joker in Gotham',
        'Tony Stark builds iron suit',
        'Love story across time',
        'Ship sinks romance blooms',
        'Dreams within dreams heist'
    ]
})

# Create TF-IDF matrix from genres
tfidf_genres = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix_genres = tfidf_genres.fit_transform(movies['genres'])

# Create TF-IDF matrix from descriptions
tfidf_desc = TfidfVectorizer(stop_words='english')
tfidf_matrix_desc = tfidf_desc.fit_transform(movies['description'])

# Combine both features (weighted)
from scipy.sparse import hstack
tfidf_combined = hstack([tfidf_matrix_genres * 0.6, tfidf_matrix_desc * 0.4])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_combined, tfidf_combined)

print("Similarity Matrix:")
print(pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title']))
```

### 4.2 Recommendation Function

```python
def get_recommendations(title, cosine_sim=cosine_sim, movies=movies, top_n=3):
    """
    Get top N similar movies
    """
    # Get index of movie
    idx = movies[movies['title'] == title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity (excluding itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return titles and scores
    recommendations = movies.iloc[movie_indices]['title'].values
    scores = [i[1] for i in sim_scores]
    
    return list(zip(recommendations, scores))

# Example
print("\nRecommendations for 'The Dark Knight':")
recs = get_recommendations('The Dark Knight')
for movie, score in recs:
    print(f"  {movie}: {score:.3f}")
```

### 4.3 User Profile-Based Recommendations

```python
# User's watched movies and ratings
user_profile = {
    'The Dark Knight': 5,
    'Iron Man': 4
}

# Build user preference vector (weighted average)
user_indices = [movies[movies['title'] == title].index[0] for title in user_profile.keys()]
user_ratings = np.array(list(user_profile.values()))

# Weighted average of item vectors
user_vector = np.average(tfidf_combined[user_indices].toarray(), 
                         axis=0, weights=user_ratings)

# Compute similarity to all movies
user_sim = cosine_similarity(user_vector.reshape(1, -1), tfidf_combined)[0]

# Get unseen movies
seen_indices = set(user_indices)
unseen_indices = [i for i in range(len(movies)) if i not in seen_indices]

# Rank unseen movies
recommendations = sorted(zip(unseen_indices, user_sim[unseen_indices]), 
                        key=lambda x: x[1], reverse=True)

print("\nPersonalized Recommendations:")
for idx, score in recommendations[:3]:
    print(f"  {movies.iloc[idx]['title']}: {score:.3f}")
```

---

## 5. Exam Preparation

### 5.1 Advantages vs Disadvantages

**Advantages:**
1. **No cold-start for items:** New movies can be recommended immediately (just need metadata)
2. **Transparency:** Clear why item was recommended
3. **No user data needed:** Works with single user
4. **Unique tastes:** Good for niche preferences

**Disadvantages:**
1. **Limited discovery:** Only recommends similar items (filter bubble)
2. **Feature engineering:** Requires good metadata
3. **New user problem:** Need some history to build profile
4. **Overspecialization:** May miss  cross-genre interests

### 5.2 Common Exam Questions

**Q1: Explain the "Cold Start Problem" in recommender systems.**

**Answer:** Three types:
1. **New User:** No history to build preferences
   - Solution: Popular items, demographic-based, explicit preferences survey
2. **New Item:** No ratings/interactions yet
   - Solution: Content-based (use metadata), manual curation
3. **New System:** No data at all
   - Solution: Popularity-based initially, gather data rapidly

**Q2: Why use cosine similarity instead of Euclidean distance?**

**Answer:**
- **Cosine:** Measures angle/orientation (direction similarity)
  - Invariant to magnitude
  - Good for sparse, high-dimensional data (text)
- **Euclidean:** Measures absolute distance
  - Sensitive to magnitude
  - Better for dense, low-dimensional data

**Example:**
```
Vector A: [1, 2, 3]      (shorter)
Vector B: [2, 4, 6]      (longer, same direction)

Cosine(A,B) = 1.0 (identical)
Euclidean(A,B) = 3.74 (different)
```

**Q3: Calculate cosine similarity between [3,4,0] and [4,3,0].**

**Solution:**
$$\text{sim} = \frac{(3)(4) + (4)(3) + (0)(0)}{\sqrt{3^2+4^2+0^2} \times \sqrt{4^2+3^2+0^2}}$$

$$= \frac{12 + 12 + 0}{\sqrt{25} \times \sqrt{25}} = \frac{24}{5 \times 5} = \frac{24}{25} = 0.96$$

### 5.3 Interview Questions

**Q: Your content-based system only recommends action movies to users who watched one action film. How to fix?**

**A:** The "filter bubble" problem. Solutions:
1. **Diversity promotion:** Add randomness/exploration
2. **Multi-factor weighting:** Don't over-weight single genre
3. **Serendipity injection:** Occasionally recommend dissimilar items
4. **Hybrid approach:** Mix with collaborative filtering
5. **Adaptive profiles:** Decay old preferences over time

**Q: How to handle multi-valued features (e.g., movie with multiple genres)?**

**A:** Several approaches:
1. **One-hot encoding:** Binary vector [Action=1, Drama=1, Romance=0, ...]
2. **Multi-hot encoding:** Same as one-hot for multi-label  
3. **Weighted encoding:** [Action=0.7, Drama=0.3, ...] based on primary/secondary
4. **Separate processing:** TF-IDF per feature type, then combine

---

## Summary

**Key Takeaways:**
- Content-based filtering uses item features to recommend similar items
- TF-IDF converts text to numerical vectors
- Cosine similarity measures vector similarity
- Works well for new items, struggles with discovery
- Hybrid approaches often best in practice

**When to Use:**
- Rich item metadata available
- New items added frequently
- Users have stable, defined preferences
- Transparency/explainability important
- Cold-start for items problematic

**Limitations:**
- Filter bubble (limited discovery)
- Requires good feature engineering
- Can't leverage wisdom of crowd
- Misses collaborative insights
