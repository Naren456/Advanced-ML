# 🎤 Real-World Interview Viva: Clustering (K-Means)

### 🏢 Walmart / Flipkart
1.  **Q:** How does K-Means++ initialization improve over random initialization?
    *   **A:** Random init can lead to bad clusters if centroids start too close. K-Means++ picks the first center randomly, but chooses subsequent centers with probability proportional to the squared distance from existing centers. This ensures centroids are well-spread, speeding up convergence.
2.  **Q:** What happens if you have clusters of different densities and sizes? Will K-Means work?
    *   **A:** **No.** K-Means assumes spherical clusters of distinct and roughly equal sizes. It will fail on crescent shapes or when one cluster is much denser than another (DBSCAN or GMM is better here).

### 🏢 Amazon
3.  **Q:** How do you evaluate Clustering if you don't have labels?
    *   **A:** **Silhouette Score** (measure of how close a point is to its own cluster vs. neighbors) or **Davies-Bouldin Index**. Ideally, Silhouette Score close to +1 is good.
