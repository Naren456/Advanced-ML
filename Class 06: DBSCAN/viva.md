# 🎤 Real-World Interview Viva: DBSCAN

### 🏢 Uber (Geospatial Data)
1.  **Q:** We want to cluster GPS pickup locations to find hotspots. K-Means gives us circular hotspots which don't match the city streets. What do you use?
    *   **A:** **DBSCAN**. It creates clusters based on density and can follow the irregular shape of road networks. It also automatically ignores outliers (GPS errors) as noise.
2.  **Q:** What is the worst-case complexity of DBSCAN?
    *   **A:** $O(N^2)$ without indexing. With a spatial index (like k-d tree), it can be $O(N \log N)$.
