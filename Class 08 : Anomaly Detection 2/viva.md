# 🎤 QA & Viva: Anomaly Detection - 2

### 🟢 Basic Questions
1.  **What is Isolation Forest?**
    *   An algorithm specifically for anomaly detection. It isolates anomalies instead of profiling normal points.

### 🟡 Intermediate Questions
2.  **How does Isolation Forest work?**
    *   It builds random trees. Anomalies are "easier to isolate" (few splits needed) because they are different. Normal points are deep in the tree (many splits). Short path length = Anomaly.
3.  **What is Local Outlier Factor (LOF)?**
    *   It compares the local density of a point to its neighbors. If a point has a much lower density than its neighbors, it is an outlier.

### 🔴 Advanced Questions
4.  **Isolation Forest vs. LOF?**
    *   **IsoForest:** Faster, scalable, works well for high dimensions (global anomalies).
    *   **LOF:** Computationally expensive but better for detecting *local* anomalies (outliers relative to a specific cluster).
