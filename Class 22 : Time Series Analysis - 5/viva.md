# 🎤 Real-World Interview Viva: ARIMA & Forecasting

### 🏢 Financial Analysis / Quant (Hedge Funds)
1.  **Q:** Why is Stationarity strictly required for ARIMA?
    *   **A:** ARIMA uses linear equations with constant coefficients. If the mean/variance changes over time (non-stationary), these coefficients would need to change, but the model assumes they are fixed.
2.  **Q:** How do you interpret the PACF plot for an AR model?
    *   **A:** For an AR($p$) process, the Partial AutoCorrelation Function (PACF) cuts off to zero after lag $p$. It helps determine the $p$ parameter.
3.  **Q:** Explain the 'I' in ARIMA.
    *   **A:** **Integrated**. It represents the number of differentiation steps ($d$) required to make the series stationary.
