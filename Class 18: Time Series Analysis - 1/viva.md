# 🎤 QA & Viva: Time Series Analysis - 1

### 🟢 Basic Questions
1.  **What is Stationarity?**
    *   A time series usually must be stationary for forecasting. Mean, Variance, and Covariance do not change over time.

### 🟡 Intermediate Questions
2.  **How do we check for Stationarity?**
    *   **Visual:** Rolling Mean/Std plot.
    *   **Statistical:** Augmented Dickey-Fuller (ADF) Test.
3.  **What is the Null Hypothesis of ADF Test?**
    *   $H_0$: The series is **Non-Stationary**.
    *   If p-value < 0.05, we reject $H_0$ $ightarrow$ Stationary.

### 🔴 Advanced Questions
4.  **How to make a series Stationary?**
    *   **Differencing:** Subtracting $Y_t - Y_{t-1}$.
    *   **Transformation:** Log or Square root (to stabilize variance).
