# 🎤 Real-World Interview Viva: Collaborative Filtering

### 🏢 Amazon / E-Commerce
1.  **Q:** User-User vs. Item-Item Filtering. Which one scales better for Amazon?
    *   **A:** **Item-Item**. Users ($N$) >>> Items ($M$) for Amazon (Millions of users, fewer distinct products). Also, item ratings are more stable (items don't "change" their nature, users change tastes).
2.  **Q:** What is the Matrix Factorization approach (e.g., in the Netflix Prize)?
    *   **A:** Decomposing the huge sparse User-Item matrix ($R$) into two lower-rank matrices: $U$ (User Factors) and $V$ (Item Factors). The dot product $U \cdot V^T$ predicts the missing ratings. 
