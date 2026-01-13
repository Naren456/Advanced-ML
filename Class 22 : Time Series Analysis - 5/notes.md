# Class 22: Time Series Analysis - 5 (Advanced Forecasting Models)

> **Core Principle:** "Modeling internal structures (Autocorrelation & Shocks) to predict the future"

---

## Table of Contents
1. [AutoRegressive Models (AR)](#1-autoregressive-models-ar)
2. [Moving Average Models (MA)](#2-moving-average-models-ma)
3. [ARMA Models](#3-arma-models)
4. [ARIMA Models](#4-arima-models)
5. [SARIMA Models](#5-sarima-models)
6. [Practical Implementation](#6-practical-implementation)
7. [Exam Preparation](#7-exam-preparation)

---

## 1. AutoRegressive Models (AR)

### 1.1 Concept & Intuition
**"History repeats itself."**
This is the fundamental assumption of AR models. It treats the current value of a time series as a linear combination of its own past values. It captures the **momentum** or "memory" of the process.

**Key Idea:** If sales were high yesterday and the day before, they are likely to be high today (assuming positive correlation).

### 1.2 Example Scenario
*   **Stock Prices:** Today's price is heavily influenced by yesterday's price. If a stock has been rising for 3 days, investor sentiment (momentum) typically pushes it up on the 4th day.
*   **Temperature:** If it is 30°C today, it is very likely to be around 30°C tomorrow, rather than jumping to 10°C. The "state" persists.

### 1.2 Mathematical Formulation (AR(p))
The value at time $t$ ($Y_t$) depends on the previous $p$ values:

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \phi_p Y_{t-p} + \epsilon_t$$

*   $Y_t$: Current value at time $t$
*   $c$: Constant (intercept)
*   $\phi_i$: Coefficient for lag $i$ (weight of the past value)
*   $Y_{t-i}$: Past value at lag $i$
*   $\epsilon_t$: White noise (random error term)
*   **Stationarity Condition:** The series **must be stationary** (constant mean/variance) for coefficients $\phi$ to be meaningful and stable.

### 1.3 Order Determination (p)
How many past days ($p$) should we look back?
*   **Tool:** **Partial Autocorrelation Function (PACF)**.
*   **Rule:** For an AR($p$) process, the PACF cuts off (drops to zero) after lag $p$, while the ACF decays gradually.
    *   *Why PACF?* It measures the direct correlation of $Y_t$ and $Y_{t-k}$ *after removing* the influence of intermediate lags.

---

## 2. Moving Average Models (MA)

### 2.1 Concept & Intuition
**"Learning from past mistakes."**
MA models predict the future based on past *forecast errors* (shocks/innovations). It assumes that a sudden shock in the system (e.g., a promo campaign or a supply glitch) has a lingering effect that dissipates over time.

### 2.2 Example Scenario
*   **Bakery Sales:** A sudden unexpected large order (shock) happens on Tuesday. This might lead to a shortage on Wednesday (negative shock) as inventory recovers, but the effect fades by Friday. The model predicts based on these recent "mistakes" or deviations.
*   **Economic Shock:** A sudden tax policy change causes a spike in spending (shock). The market corrects itself over the next few months as the shock dampens.

### 2.2 Mathematical Formulation (MA(q))
The value at time $t$ depends on the mean and the previous $q$ error terms:

$$Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$$

*   $\mu$: Mean of the series
*   $\epsilon_t$: Current error (shock)
*   $\theta_i$: Coefficient for past error at lag $i$
*   $\epsilon_{t-i}$: Past error (shock) at lag $i$

### 2.3 Order Determination (q)
How many past shocks ($q$) affect today?
*   **Tool:** **Autocorrelation Function (ACF)**.
*   **Rule:** For an MA($q$) process, the ACF cuts off after lag $q$, while the PACF decays gradually.

---

## 3. ARMA Models

### 3.1 Concept
**"Best of both worlds."**
Real-world data often has both momentum (AR) and shock-response (MA) characteristics. ARMA(p, q) combines these to provide a parsimonious model (fewer parameters than pure AR or MA).

### 3.2 Formulation
$$Y_t = c + \underbrace{\sum_{i=1}^p \phi_i Y_{t-i}}_{AR} + \underbrace{\sum_{j=1}^q \theta_j \epsilon_{t-j}}_{MA} + \epsilon_t$$

*   **Assumption:** The series must be **Stationary**.

---

## 4. ARIMA Models

### 4.1 Integration (I) for Non-Stationarity
**"Stationarizing the data internally."**
Most real data (like stock prices or sales) has a trend and is **non-stationary**.
*   **ARIMA(p, d, q):**
    *   **AR (p):** AutoRegressive term.
    *   **I (d):** Integrated term (Differencing order). The number of times raw data needs differencing to become stationary.
    *   **MA (q):** Moving Average term.

### 4.2 Why ARIMA?
Instead of manually differencing the data, fitting a model, and then manually integrating (cumulative sum) the forecast, ARIMA handles this pipeline automatically.

**Workflow:**
1.  **Identify d:** Use ADF test to find d such that $Y_t^{(d)}$ is stationary.
2.  **Identify p, q:** Use PACF/ACF on the *differenced* data.
3.  **Fit Model:** Pass raw data to ARIMA with $(p, d, q)$.
4.  **Forecast:** Output is automatically in the original scale.

### 4.3 Example Scenario
*   **GDP Growth:** A country's GDP is continuously growing (non-stationary trend). To predict next year's GDP, we look at the *change* in GDP from year to year (differencing to make it stationary), model that change using ARMA, and then add it back to the current level.

---

## 5. SARIMA Models

### 5.1 Seasonality (S)
**"Capturing repeating cycles."**
ARIMA fails if there's a strong seasonal pattern (e.g., repeating every 12 months). **S**easonal **ARIMA** adds parameters specifically for seasonal lags ($m, 2m, 3m...$).

### 5.2 Notation
**SARIMA(p, d, q)(P, D, Q)m**
*   **(p, d, q):** Non-seasonal components (Trend/Short-term).
*   **(P, D, Q):** Seasonal components (Cycle/Long-term).
*   **m:** Seasonal period (e.g., 12 for monthly, 4 for quarterly).

**Example:** SARIMA(1, 1, 1)(1, 1, 1)12
*   Predicts $Y_t$ using last month's value (AR1) AND last year's value (Seasonal AR1).
*   Differences data month-to-month ($d=1$) AND year-over-year ($D=1$).

### 5.3 Example Scenario
*   **Ice Cream Sales:**
    *   **Trend:** Sales are generally increasing every year due to population growth (ARIMA part).
    *   **Seasonality:** Sales always peak in June/July (Summer) and drop in December (Winter). SARIMA captures this yearly cycle ($m=12$) on top of the general growth.

---

## 6. Practical Implementation

We use `statsmodels.tsa.statespace.sarimax.SARIMAX` for all models (AR, MA, ARMA, ARIMA) by setting appropriate orders.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- 1. Data Preparation ---
# Ideally, check stationarity first. 
# For ARIMA/SARIMA, raw data can be used if 'd' or 'D' is specified.

# --- 2. Model Fitting ---

# A. AR(1) Model: (p=1, d=0, q=0)
# Use stationary data (train_x_st)
model_ar = SARIMAX(train_x_st['Sales'], order=(1, 0, 0))
res_ar = model_ar.fit(disp=False)

# B. MA(2) Model: (p=0, d=0, q=2)
# Use stationary data
model_ma = SARIMAX(train_x_st['Sales'], order=(0, 0, 2))
res_ma = model_ma.fit(disp=False)

# C. ARMA(3, 3) Model: (p=3, d=0, q=3)
# Use stationary data
model_arma = SARIMAX(train_x_st['Sales'], order=(3, 0, 3))
res_arma = model_arma.fit(disp=False)

# D. ARIMA(3, 1, 3) Model: (p=3, d=1, q=3)
# Use RAW data (train_x). The model handles differencing (d=1).
model_arima = SARIMAX(train_x['Sales'], order=(3, 1, 3))
res_arima = model_arima.fit(disp=False)

# E. SARIMA Model: (p,d,q) + (P,D,Q,m)
# Use RAW data. Handles both regular and seasonal differencing.
model_sarima = SARIMAX(train_x['Sales'], 
                       order=(3, 1, 3), 
                       seasonal_order=(1, 1, 1, 12))
res_sarima = model_sarima.fit(disp=False)

# --- 3. Forecasting & Evaluation ---

def evaluate_model(model_res, test_data, steps=12):
    # Forecast
    pred = model_res.forecast(steps=steps)
    
    # If using AR/MA/ARMA on manually differenced data, 
    # you might need to inverse transform here (e.g., cumsum).
    # ARIMA/SARIMA output is already in correct scale.
    
    # Visualization
    plt.figure(figsize=(10,6))
    plt.plot(test_data, label='Actual')
    plt.plot(pred, label='Forecast', linestyle='--')
    plt.legend()
    plt.show()
    
    # Metrics
    performance(test_data, pred) # Custom function for RMSE/MAPE

# Example call
evaluate_model(res_sarima, test_x['Sales'])
```

---

## 7. Exam Preparation

### **Key Concepts to Review:**
1.  **Stationarity:** Why is it strictly required for AR/MA terms? (Stability of $\phi, \theta$ coefficients).
2.  **ACF vs PACF:**
    *   **AR(p) identification:** Look at PACF (cuts off at p).
    *   **MA(q) identification:** Look at ACF (cuts off at q).
3.  **Model Selection:**
    *   Why choose ARIMA over ARMA? (To handle trends automatically).
    *   Why choose SARIMA over ARIMA? (To handle seasonality).
4.  **Residual Analysis:**
    *   Ideally, residuals ($\epsilon_t$) should be white noise (no pattern, mean 0). If patterns remain in residuals, the model is under-fitted.

### **Common Questions:**
*   *If ACF decays slowly and PACF cuts off at lag 2, what model is suggested?* -> **AR(2)**.
*   *What does the 'd' in ARIMA(1,1,1) represent?* -> **1st order Differencing**.
*   *Can SARIMA handle data with both yearly seasonality and a linear trend?* -> **Yes**, by setting $d=1$ and appropriate seasonal parameters.
