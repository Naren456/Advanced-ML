# Class 21: Time Series Analysis - 4

## 1. Overview
This class delves deeper into the statistical foundations required for advanced forecasting models like ARIMA. The central theme is **Stationarity** and the tools used to diagnose and model time series dependencies: **ACF** (AutoCorrelation Function) and **PACF** (Partial AutoCorrelation Function).

## 2. Stationarity
A time series is considered **Stationary** if its statistical properties do not change over time. This is a crucial assumption for many time series forecasting methods (like ARIMA), which rely on the idea that "what happened in the past will repeat in the future" in a statistical sense.

### Key Characteristics of a Stationary Series:
*   **Constant Mean:** The average value of the series remains roughly constant over time. There is no long-term trend.
*   **Constant Variance:** The spread (variability) of the data around the mean is consistent over time. It doesn't expand or contract (heteroscedasticity).
*   **Constant Covariance:** The relationship (correlation) between two observations depends only on the time lag between them, not on the absolute time they were observed.

### Types of Non-Stationarity:
*   **Trend:** A long-term increase or decrease in the data.
*   **Seasonality:** Repeating patterns at fixed intervals.

### Testing for Stationarity
1.  **Visual Inspection:** Plotting the rolling mean and variance to check for stability.
2.  **Augmented Dickey-Fuller (ADF) Test:** A rigorous statistical test.
    *   **Null Hypothesis ($H_0$):** The time series is **Non-Stationary** (has a unit root).
    *   **Alternative Hypothesis ($H_1$):** The time series is **Stationary**.
    *   **Interpretation:**
        *   If **p-value < 0.05**: Reject $H_0$. The series is Stationary.
        *   If **p-value > 0.05**: Fail to reject $H_0$. The series is Non-Stationary.

## 3. Converting Non-Stationary to Stationary
If a series is non-stationary, we must transform it before modeling.

### Techniques:
1.  **De-trending:**
    *   **Differencing:** Calculating the difference between consecutive observations: $y'_t = y_t - y_{t-1}$. This removes the linear trend (constant mean assumption).
    *   **Transformation:** Applying Log or Square Root transformations to stabilize non-constant variance.
2.  **De-seasonalizing:**
    *   **Seasonal Differencing:** Subtracting the value from the same season in the previous cycle (e.g., $y'_t = y_t - y_{t-12}$ for monthly data).

## 4. ACF and PACF
These plots are the primary tools for identifying the component orders ($p, d, q$) for ARIMA models.

### AutoCorrelation Function (ACF)
*   **Definition:** Measures the correlation between the time series and its lagged versions (e.g., $Y_t$ vs $Y_{t-1}$, $Y_t$ vs $Y_{t-2}$, etc.).
*   **Includes:** Both direct and indirect effects. For example, $Y_{t-2}$ affects $Y_{t-1}$, which in turn affects $Y_t$. The ACF at lag 2 captures the total correlation.
*   **Usage:** Helps identify the order of the **MA (Moving Average)** term ($q$).
    *   In an MA($q$) process, the ACF cuts off after lag $q$.

### Partial AutoCorrelation Function (PACF)
*   **Definition:** Measures the correlation between $Y_t$ and $Y_{t-k}$ that is **not** explained by the intermediate lags ($Y_{t-1}, ..., Y_{t-k+1}$). It isolates the direct correlation.
*   **Usage:** Helps identify the order of the **AR (AutoRegressive)** term ($p$).
    *   In an AR($p$) process, the PACF cuts off after lag $p$.

## 5. Summary Table for Model Identification

| Model Type | ACF (Autocorrelation) | PACF (Partial Autocorrelation) |
| :--- | :--- | :--- |
| **AR($p$)** | Decays gradually (Geometric/Sinusoidal) | **Cuts off** after lag $p$ |
| **MA($q$)** | **Cuts off** after lag $q$ | Decays gradually |
| **ARMA($p, q$)** | Decays gradually | Decays gradually |
