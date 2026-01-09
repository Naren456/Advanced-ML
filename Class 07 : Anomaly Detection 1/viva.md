# 🎤 Real-World Interview Viva: Anomaly Detection

### 🏢 FinTech (Visa/Mastercard)
1.  **Q:** In fraud detection, most data is normal (99.9%). How do you handle this extreme imbalance?
    *   **A:** Don't focus on Accuracy (it will be 99.9% by just guessing 'safe'). Focus on **Recall** (catching all frauds) and **Precision** (minimizing false alarms). Use algorithms explicitly designed for outliers like **Isolation Forest** or **One-Class SVM**.
2.  **Q:** Why does Z-Score fail in some distributions?
    *   **A:** Z-Score assumes a **Gaussian (Normal) Distribution**. If the data is bi-modal or skewed, Z-score thresholds (like >3) are invalid.

### 🏢 Netflix (System Health)
3.  **Q:** How does Isolation Forest work differently from other methods?
    *   **A:** Most methods try to model "Normal" behavior. Isolation Forest explicitly tries to isolate points. Anomalies are "easiest" to isolate (require fewer random cuts in a tree), so they have shorter path lengths.
