# Notes: Recommender Systems - 2 (Class 12)

## Overview
This class continues the topic of Recommender Systems, specifically focusing on **Market Basket Analysis** and **Association Rules** using the **Apriori Algorithm**. Ideally, this serves as a practical implementation or a deeper dive into the concepts introduced in Class 11.

## Key Concepts Refresher
*   **Market Basket Analysis:** techniques to uncover associations between items.
*   **Apriori Algorithm:** A classic algorithm for learning association rules. It operates on the principle that if an itemset is frequent, then all of its subsets must also be frequent.
*   **Metrics:**
    *   **Support:** Popularity of an itemset.
    *   **Confidence:** Likelihood of Y given X.
    *   **Lift:** Strength of association ( > 1 means likely together).

## Code Implementation
The notebook `ML_RS_Apriori_and_Association_Rules.ipynb` demonstrates the end-to-end process:

### 1. Data Prep
*   **Dataset:** `Online_Retail.csv`
*   **Cleaning:**
    *   Remove records with negative `Quantity` (returns/cancellations).
    *   Remove records with negative `UnitPrice`.
    *   Filter for a specific country (e.g., "United Kingdom") to keep the analysis manageable and relevant.
*   **Transformation:**
    *   Create a clean DataFrame with `InvoiceNo`, `Description`, and `Quantity`.
    *   **Pivot Table:** Transform to a format where rows are Invoices and columns are Products.
    *   **One-Hot Encoding:** Convert quantities to 0 or 1 (bought or not bought).

### 2. Apriori Algorithm
*   **Library:** `mlxtend` (Machine Learning Extensions).
*   **Process:**
    *   `apriori(df, min_support=0.03, use_colnames=True)`: Finds frequent itemsets with at least 3% support.
    *   The result is a DataFrame of itemsets and their support values.

### 3. Association Rules
*   **Process:**
    *   `association_rules(frequent_itemsets, metric="lift", min_threshold=1)`: Generates rules from the frequent itemsets.
    *   Filters for rules where `lift >= 1` (meaning meaningful positive association).
*   **Result:** A DataFrame containing antecedents, consequents, support, confidence, and lift for each rule.

## Insights
*   The analysis helps identifying which products are frequently bought together (e.g., "Pink Glazed Candle" and "Red Hanging Heart").
*   These insights can be used for:
    *   **Bundling:** Selling items together at a discount.
    *   **Store Layout:** Placing associated items near each other.
    *   **Recommendation:** Suggesting "You might also like..."
