# 🎤 QA & Viva: Time Series Analysis - 3 (Smoothing)

### 🟢 Basic Questions
1.  **What is "Smoothing"?**
    *   Removing noise to reveal interesting trends.
2.  **Simple Moving Average (SMA)?**
    *   Avg of last $k$ periods. Good for stable data, lags behind trends.

### 🟡 Intermediate Questions
3.  **Holt-Winters Method (Triple Exponential Smoothing)?**
    *   Handles **Level**, **Trend**, and **Seasonality**.
    *   Parameters: $lpha$ (Level), $eta$ (Trend), $\gamma$ (Seasonality).

### 🔴 Advanced Questions
4.  **Why Exponential Smoothing over Moving Average?**
    *   Exponential Smoothing gives **more weight to recent observations**, whereas Moving Average treats old and new data inside the window equally.
