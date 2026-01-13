# Class 20: Time Series Analysis - 3

> **Core Principle:** "Smoothing out noise to reveal the signal."

---

## Table of Contents
1. [Review of Previous Methods](#1-review-of-previous-forecasting-methods)
2. [Introduction to Smoothing](#2-introduction-to-smoothing-methods)
3. [Moving Average](#3-moving-average)
4. [Simple Exponential Smoothing (SES)](#4-simple-exponential-smoothing-ses)
5. [Double Exponential Smoothing (Holt's Method)](#5-double-exponential-smoothing-des--holts-method)
6. [Triple Exponential Smoothing (Holt-Winters Method)](#6-triple-exponential-smoothing-tes--holt-winters-method)
7. [Comparison of Methods](#7-comparison-of-methods)
8. [Preview of Next Topics](#8-preview-of-upcoming-topics)
9. [Exam Preparation](#9-advice-for-exam-preparation)

---

## 1. Review of Previous Forecasting Methods

In the previous class, we established baseline models. While simple, they provide crucial benchmarks.

*   **Simple Mean:** Forecasts the average of all past observations. Good for stationary Series with no trend or seasonality.
*   **Naive Approach:** Forecasts the last observed value ($Y_{t+1} = Y_t$). Works well for random walks.
*   **Seasonal Naive:** Forecasts the value from the same season in the previous cycle ($Y_{t+1} = Y_{t-m}$). Great for strong seasonality.
*   **Drift Method:** Extends the line between the first and last observation. Captures linear trend but ignores intermediate fluctuations.

**Why move beyond them?**
Most real-world data contains a mix of **Trend**, **Seasonality**, and **Noise**. Baselines typically capture only one (or none) effectively. We need methods that can separate signal from noise.

---

## 2. Introduction to Smoothing Methods

**Concept:** 
Smoothing methods are based on the idea of **weighted averages**. Instead of taking a simple average (equal weights) or just the last value (100% weight on $t$, 0% on others), we assign **decreasing weights** to older observations.

**Weighing The Past:**
*   **Recent data** is more relevant $\rightarrow$ Higher weight.
*   **Older data** is less relevant $\rightarrow$ Lower weight.

**The Family of Exponential Smoothing:**
1.  **Simple (SES):** Level only (No trend, No seasonality).
2.  **Double (Holt's):** Level + Trend (No seasonality).
3.  **Triple (Holt-Winters):** Level + Trend + Seasonality.

---

## 3. Moving Average

**Concept:**
Calculates the average of a fixed window (e.g., last 12 months) that "moves" forward. It smooths out short-term fluctuations to highlight the underlying trend.

**Equation:**
$$\hat{y}_{t+1} = \frac{Y_t + Y_{t-1} + ... + Y_{t-k+1}}{k}$$

**Python Implementation:**

```python
# Moving Average Forecast (Window = 12 months)
train_len = 212
test_len = 24
window_size = 12

# Create a copy for predictions
y_hat_ma = test_x.copy()

# Calculate rolling mean on training data
# We take the last available rolling mean as the forecast for the future
last_rolling_mean = train_x['Sales'].rolling(window_size).mean().iloc[-1]
y_hat_ma['moving_avg_forecast'] = last_rolling_mean

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(train_x['Sales'], label='Train')
plt.plot(test_x['Sales'], label='Test')
plt.plot(y_hat_ma['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.title('Moving Average Forecast')
plt.show()

# Performance
performane(test_x['Sales'], y_hat_ma['moving_avg_forecast'])
```

**Observation:**
*   **Performance:** MAPE ~ 9.4% (Context dependent).
*   **Limitation:** It produces a **flat line** forecast for the future horizon. It completely fails to capture dynamic trends or seasonality *after* the window ends. It lags behind the trend.

---

## 4. Simple Exponential Smoothing (SES)

**Concept:**
Instead of a hard cutoff like Moving Average, SES uses **all** past data but assigns exponentially decreasing weights. It defines a "Level" ($\ell_t$) which evolves.

**Equation:**
$$ \ell_t = \alpha Y_t + (1 - \alpha) \ell_{t-1} $$
$$ \hat{y}_{t+1} = \ell_t $$

*   $\alpha$ (alpha): Smoothing parameter for **Level**. $0 < \alpha < 1$.
*   High $\alpha$: Trust recent data more (reactive).
*   Low $\alpha$: Trust past average more (smooth).

**Python Implementation:**

```python
from statsmodels.tsa.api import SimpleExpSmoothing

# Fit SES Model
model_ses = SimpleExpSmoothing(train_x['Sales']).fit(smoothing_level=0.2, optimized=False)
y_hat_ses = test_x.copy()
y_hat_ses['ses_forecast'] = model_ses.forecast(test_len)

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(train_x['Sales'], label='Train')
plt.plot(test_x['Sales'], label='Test')
plt.plot(y_hat_ses['ses_forecast'], label='SES Forecast')
plt.legend(loc='best')
plt.title('Simple Exponential Smoothing (Alpha=0.2)')
plt.show()

# Performance
performane(test_x['Sales'], y_hat_ses['ses_forecast'])
```

**Observation:**
*   **Performance:** MAPE ~ 11.3%.
*   **Limitation:** Like MA, it still produces a **flat forecast**. It assumes the series will stay at the last estimated "level". Suitable only for data without trend or seasonality.

---

## 5. Double Exponential Smoothing (DES) / Holt's Method

**Concept:**
Adds a second equation to handle **Trend** ($b_t$). Now we forecast the Level + Trend.

**Equations:**
1.  **Level:** $\ell_t = \alpha Y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})$
2.  **Trend:** $b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1}$
3.  **Forecast:** $\hat{y}_{t+h} = \ell_t + h b_t$

*   $\beta$ (beta): Smoothing parameter for **Trend**.

**Python Implementation:**

```python
from statsmodels.tsa.api import ExponentialSmoothing

# Fit Holt's Model (Trend='add' for additive trend)
# seasonal_periods=12 helps initialize, though Holt's doesn't model seasonality explicitly
model_holt = ExponentialSmoothing(train_x['Sales'], seasonal_periods=12, trend='add', seasonal=None).fit()

y_hat_holt = test_x.copy()
y_hat_holt['holt_forecast'] = model_holt.forecast(test_len)

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(train_x['Sales'], label='Train')
plt.plot(test_x['Sales'], label='Test')
plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s Forecast')
plt.legend(loc='best')
plt.title('Double Exponential Smoothing (Holt\'s Method)')
plt.show()

# Performance
performane(test_x['Sales'], y_hat_holt['holt_forecast'])
```

**Observation:**
*   **Performance:** MAPE ~ 8.9% (Improvement).
*   **Result:** The forecast is a **sloped line**. It captures the general upward/downward direction but misses the wave-like seasonal pattern.

---

## 6. Triple Exponential Smoothing (TES) / Holt-Winters Method

**Concept:**
The "Gold Standard" of exponential smoothing. Adds a third component for **Seasonality** ($s_t$).

**Equations:**
1.  **Level:** $\ell_t = \alpha (Y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$
2.  **Trend:** $b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1}$
3.  **Seasonality:** $s_t = \gamma (Y_t - \ell_t - b_{t-1}) + (1 - \gamma)s_{t-m}$
4.  **Forecast:** $\hat{y}_{t+h} = \ell_t + h b_t + s_{t+h-m}$

*   $\gamma$ (gamma): Smoothing parameter for **Seasonality**.
*   **Additive vs. Multiplicative:**
    *   **Additive:** Seasonal peaks stay constant in size (+$S_t$).
    *   **Multiplicative:** Seasonal peaks grow with the trend ($\times S_t$).

**Python Implementation (Additive):**

```python
# Holt-Winters Additive
model_hw_add = ExponentialSmoothing(train_x['Sales'], seasonal_periods=12, trend='add', seasonal='add').fit()

y_hat_hw = test_x.copy()
y_hat_hw['hw_add_forecast'] = model_hw_add.forecast(test_len)

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(train_x['Sales'], label='Train')
plt.plot(test_x['Sales'], label='Test')
plt.plot(y_hat_hw['hw_add_forecast'], label='HW Additive Forecast')
plt.legend(loc='best')
plt.title('Holt-Winters Additive')
plt.show()

# Performance
performane(test_x['Sales'], y_hat_hw['hw_add_forecast'])
```

**Python Implementation (Multiplicative):**

```python
# Holt-Winters Multiplicative
# Use when seasonal variance increases with the level of the series
model_hw_mul = ExponentialSmoothing(train_x['Sales'], seasonal_periods=12, trend='add', seasonal='mul').fit()

y_hat_hw['hw_mul_forecast'] = model_hw_mul.forecast(test_len)

# Visualization
plt.figure(figsize=(12, 4))
plt.plot(train_x['Sales'], label='Train')
plt.plot(test_x['Sales'], label='Test')
plt.plot(y_hat_hw['hw_mul_forecast'], label='HW Multiplicative Forecast')
plt.legend(loc='best')
plt.title('Holt-Winters Multiplicative')
plt.show()

# Performance
performane(test_x['Sales'], y_hat_hw['hw_mul_forecast'])
```

**Observation:**
*   **Performance:** MAPE drops significantly (e.g., ~ 4-5%).
*   **Result:** The forecast captures **Level, Trend, AND Seasonality**. It looks like a realistic continuation of the history.

---

## 7. Comparison of Methods

| Method | Components | Forecast Shape | Typical MAPE |
| :--- | :--- | :--- | :--- |
| **Moving Average** | None (Window Avg) | Flat Line | ~9.4% |
| **Simple Exp Smo (SES)** | Level | Flat Line | ~11.3% |
| **Holt's (DES)** | Level + Trend | Sloped Line | ~8.9% |
| **Holt-Winters (Additive)** | L + T + Seasonality | Constant Waves | ~5.1% |
| **Holt-Winters (Multiplicative)** | L + T + Seasonality | Growing Waves | ~4.6% |

> **Key Learning:** Adding the correct components (Trend and Seasonality) drastically reduces error.

---

## 8. Preview of Upcoming Topics

We have mastered Smoothing. Next, we enter the world of statistical correlation models using **ARIMA**.

1.  **Stationarity:** The most critical prerequisite for ARIMA.
    *   Mean, Variance, and Covariance must be constant over time.
    *   We will test this using the **ADF Test (Augmented Dickey-Fuller)**.
2.  **Differencing:** How to make non-stationary data stationary.
3.  **ARIMA:** AutoRegressive Integrated Moving Average.
    *   **AR:** Regressing on past values (lags).
    *   **I:** Integrated (Differencing).
    *   **MA:** Moving Average (modeling error terms).

---

## 9. Advice for Exam Preparation

*   **Understand the Alpha/Beta/Gamma:** Know what happens if they are 0 or 1. (High = fast reaction/noisy, Low = slow reaction/smooth).
*   **Identify the Plot:** Be able to look at a forecast plot and say "That is SES because it's flat" or "That is Holt's because it's a straight line."
*   **Code Syntax:** Remember the key library: `from statsmodels.tsa.api import ExponentialSmoothing`.
*   **Evaluation:** MAPE is often the most interpretable metric for business stakeholders.
