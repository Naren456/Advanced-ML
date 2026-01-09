# Notes: Recommender Systems - 1 (Class 11)

## Key Concepts

### Recommender Systems
Systems that predict user preferences and suggest items they might be interested in.
*   **Examples:** Netflix (Movies), YouTube (Videos), Amazon (Products), Instagram (Content), various dating apps.
*   **Business Value:**
    *   Improve User Experience (reduce search time).
    *   Increase Revenue/Profit (cross-sell/upsell).
    *   User Retention (reduce churn).

### Types of Recommender Systems
1.  **Popularity Based:**
    *   Recommends most bought/viewed/trending items.
    *   **Pros:** Simple, good for new users (Cold Start problem).
    *   **Cons:** No personalization.
2.  **Content-Based Filtering:**
    *   Recommends items similar to those a user liked in the past (e.g., if you liked "The Office", recommend "Parks and Rec").
    *   Relies on item features/tags.
3.  **Collaborative Filtering:**
    *   Recommends items based on the behavior of similar users (e.g., "Users who liked X also liked Y").
    *   Leverages user-item interactions.
4.  **Hybrid:** Combination of the above.

## Market Basket Analysis
A technique to identify associations between items purchased together.
*   **Goal:** Find relationships like "If a customer buys Bread, they are likely to buy Butter".
*   **Applications:** Product placement, Cross-selling, Upselling, Combo offers.

### Association Rules Metrics
*   **Support:** Probability of an item (or itemset) occurring in a transaction.
    *   $Support(X) = \frac{\text{Transactions with X}}{\text{Total Transactions}}$
*   **Confidence:** Probability of buying item Y given that item X is bought.
    *   $Confidence(X \rightarrow Y) = \frac{Support(X \cup Y)}{Support(X)}$
*   **Lift:** The ratio of observed support to expected support if X and Y were independent.
    *   $Lift(X \rightarrow Y) = \frac{Confidence(X \rightarrow Y)}{Support(Y)} = \frac{Support(X \cup Y)}{Support(X) \times Support(Y)}$
    *   **Lift > 1:** Positive correlation (likely bought together).
    *   **Lift = 1:** Independent (no relationship).
    *   **Lift < 1:** Negative correlation (unlikely to be bought together).

### Apriori Algorithm
An algorithm to find frequent itemsets and generate association rules.
1.  Calculate support for individual items.
2.  Filter items below a minimum support threshold.
3.  Form 2-item combinations from frequent patterns and check support.
4.  Repeat for larger itemsets until no more frequent itemsets can be found.
5.  Generate rules from frequent itemsets based on Confidence and Lift.

## Code Implementation
The notebook explores Market Basket Analysis using the `Online_Retail.csv` dataset.

### Steps:
1.  **Data Loading & Cleaning:**
    *   Filter valid quantities (`Quantity > 0`) and prices.
    *   Focus analysis on "United Kingdom" transactions.
2.  **Data Transformation:**
    *   Group by `InvoiceNo` and `Description`.
    *   Pivot/Unstack the data to create a transaction matrix (Rows: Invoices, Cols: Products).
    *   **One-Hot Encoding:** Convert quantities to binary (1 if present, 0 otherwise).
3.  **Association Rule Mining (Intended):**
    *   Use `mlxtend.frequent_patterns.apriori` to find frequent itemsets (e.g., `min_support=0.03`).
    *   Use `mlxtend.frequent_patterns.association_rules` to generate rules (e.g., filtering by `lift >= 1`).
    *(Note: The notebook contains some execution errors in the final steps but outlines this standard process.)*
