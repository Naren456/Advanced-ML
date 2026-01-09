# Notes: Recommender Systems - 3 (Class 13)

## Evolution of Recommender Systems
1.  **Pre-2007:** Similarity Based (Content-based, Collaborative Filtering).
2.  **2007-2015:** **Matrix Factorization** (Popularized by the Netflix Prize).
3.  **Post-2015:** Deep Learning based systems.

## Key Concepts

### 1. Collaborative Filtering (CF)
*   **Core Idea:** Recommendations are based on the interactions between users and items (Ratings, Clicks, Purchases).
*   **The Matrix:**
    *   Data is represented as an $n \times m$ matrix (Users $\times$ Items).
    *   **Sparse Matrix:** Most entries are empty/unknown because a single user interacts with a tiny fraction of total items.
*   **Types:**
    *   **User-User CF:** Finds users similar to the target user (e.g., "People like you also liked...").
    *   **Item-Item CF:** Finds items similar to what the user liked (e.g., Amazon's "Customers who bought this also bought...").
*   **Similarity Measures:** Cosine Similarity is commonly used to calculate closeness between vectors.

### 2. Content-Based Filtering (CB)
*   **Core Idea:** Recommendations are based on the attributes (metadata) of users and items.
*   **Metadata:**
    *   **User:** Age, Gender, Location, Device, etc.
    *   **Item:** Genre, Description, Price, Category, etc.
*   **Mechanism:**
    *   Can be modeled as a **Classification** problem (Will the user click? Yes/No) or **Regression** problem (Predict Rating).
*   **Pros:** Solves the **Cold Start Problem** (where new users/items lack interaction history).

### 3. Cold Start Problem
*   **Issue:** CF systems cannot recommend to new users (no history) or recommend new items (no interactions).
*   **Solution:** Use Content-Based Filtering (utilize metadata) or Popularity-based recommendations initially.

### 4. Hybrid Systems
*   Combine Collaborative Filtering (for personalization based on behavior) and Content-Based Filtering (to handle new items/users and use metadata).

### 5. Matrix Factorization
*   A powerful technique to reduce dimensionality and identify latent features underlying the interactions.
*   Decomposes the sparse User-Item matrix into lower-dimensional matrices (User Factors and Item Factors).
*   Crucial for scalable & accurate recommendations (Netflix Prize winner).
