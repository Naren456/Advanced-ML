# Class 18: Time Series Analysis - Stationarity and Baseline Forecasting

> **Core Principle:** "Making data stationary before modeling"

---

## Table of Contents
1. [Stationarity](#1-stationarity)
2. [Testing for Stationarity](#2-testing)
3. [Making Series Stationary](#3-transformation)
4. [Baseline Forecasting Methods](#4-baselines)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. Stationarity

### 1.1 Definition

**Stationary Series:** Statistical properties remain constant over time

**Three Requirements:**
1. **Constant mean:** $E[Y_t] = \mu$ for all $t$
2. **Constant variance:** $\text{Var}(Y_t) = \sigma^2$ for all $t$
3. **Constant covariance:** $\text{Cov}(Y_t, Y_{t+k})$ depends only on lag $k$, not time $t$

### 1.2 Visual Examples

**Stationary:**
```
Sales
  |  ~~~~~~~~~~~~~~~~~~~~~~~~
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~
  |~~~~~~~~~~~~~~~~~~~~~~~~~~~
  |___________________________
  Time
(Mean and variance constant)
```

**Non-Stationary (Trend):**
```
Sales
  |                    /
  |                  /
  |                /
  |              /
  |            /
  |__________/_______________
  Time
(Mean changes over time)
```

**Non-Stationary (Changing Variance):**
```
Sales
  |              /\/\/\/\/\
  |            /\/\/\
  |          /\/\
  |        /\
  |______/___________________
  Time
(Variance increases over time)
```

### 1.3 Why Stationarity Matters

**Problem:** Models like ARIMA assume constant statistical properties

**If non-stationary:**
- Parameters estimated on past won't work on future
-Different time periods have different behavior
- Forecasts unreliable

**Analogy:** Trying to predict tomorrow's weather using last month's weather patterns when seasons are changing

---

## 2. Testing

### 2.1 Visual Test - Rolling Statistics

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_rolling_statistics(series, window=12):
    """
    Plot rolling mean and std to check stationarity visually
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original vs Rolling Mean
    axes[0].plot(series, label='Original', alpha=0.6)
    axes[0].plot(rolling_mean, label='Rolling Mean', color='red', linewidth=2)
    axes[0].legend()
    axes[0].set_title('Original Series vs Rolling Mean')
    axes[0].grid(True, alpha=0.3)
    
    # Rolling Std
    axes[1].plot(rolling_std, label='Rolling Std', color='black', linewidth=2)
    axes[1].legend()
    axes[1].set_title('Rolling Standard Deviation')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Simple check
    mean_var = rolling_mean.var()
    std_var = rolling_std.var()
    
    print(f"Variance of rolling mean: {mean_var:.4f}")
    print(f"Variance of rolling std: {std_var:.4f}")
    
    if mean_var < 0.1 * series.var() and std_var < 0.1 * series.var():
        print("Visual check: Likely STATIONARY")
    else:
        print("Visual check: Likely NON-STATIONARY")

# Usage
df = pd.read_excel('mobilesales.xlsx', parse_dates=['DATE'], index_col='DATE')
plot_rolling_statistics(df['Sales'])
```

### 2.2 ADF Test (Augmented Dickey-Fuller)

**Statistical Test for Stationarity**

**Hypotheses:**
- $H_0$: Series has unit root (NON-stationary)
- $H_1$: Series is stationary

**Decision Rule:**
- If p-value < 0.05: Reject $H_0$ → **Stationary**
- If p-value ≥ 0.05: Fail to reject $H_0$ → **Non-stationary**

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    """
    Perform Augmented Dickey-Fuller test
    """
    result = adfuller(series.dropna())
    
    print(f'\n{"="*60}')
    print(f'ADF Test Results for {name}')
    print(f'{"="*60}')
    print(f'Test Statistic     : {result[0]:.6f}')
    print(f'p-value            : {result[1]:.6f}')
    print(f'Lags Used          : {result[2]}')
    print(f'Number of Observations: {result[3]}')
    print(f'\nCritical Values:')
    for key, value in result[4].items():
        print(f'   {key:>5s}: {value:.4f}')
    
    # Interpretation
    print(f'\n{"="*60}')
    if result[1] <= 0.05:
        print(f'Result: STATIONARY (Reject H0)')
        print(f'Reason: p-value ({result[1]:.6f}) < 0.05')
        return True
    else:
        print(f'Result: NON-STATIONARY (Fail to reject H0)')
        print(f'Reason: p-value ({result[1]:.6f}) >= 0.05')
        return False

# Test original series
is_stationary = adf_test(df['Sales'], 'Mobile Sales')
```

### 2.3 KPSS Test (Complementary)

**KPSS = Kwiatkowski-Phillips-Schmidt-Shin**

**Opposite hypotheses from ADF:**
- $H_0$: Series is stationary
- $H_1$: Series has unit root

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, name=''):
    """
    Perform KPSS test
    """
    result = kpss(series.dropna(), regression='c')
    
    print(f'\nKPSS Test Results for {name}')
    print(f'Test Statistic: {result[0]:.6f}')
    print(f'p-value       : {result[1]:.6f}')
    
    if result[1] >= 0.05:
        print('Result: STATIONARY (Fail to reject H0)')
        return True
    else:
        print('Result: NON-STATIONARY (Reject H0)')
        return False

# Combined test
adf_stationary = adf_test(df['Sales'], 'Sales')
kpss_stationary = kpss_test(df['Sales'], 'Sales')

if adf_stationary and kpss_stationary:
    print("\n✓ Both tests agree: STATIONARY")
elif not adf_stationary and not kpss_stationary:
    print("\n✗ Both tests agree: NON-STATIONARY")
else:
    print("\n? Tests disagree: Check visually")
```

---

## 3. Transformation

### 3.1 Differencing

**First Difference:**
$$Y'_t = Y_t - Y_{t-1}$$

```python
# Apply differencing
df['Sales_diff'] = df['Sales'].diff()

# Test stationarity
adf_test(df['Sales_diff'].dropna(), 'First Differenced Sales')

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df['Sales'])
axes[0].set_title('Original Series')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['Sales_diff'])
axes[1].set_title('After First Differencing')
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Second Difference (if needed):**
$$Y''_t = Y'_t - Y'_{t-1} = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})$$

```python
df['Sales_diff2'] = df['Sales_diff'].diff()
adf_test(df['Sales_diff2'].dropna(), 'Second Differenced Sales')
```

### 3.2 Seasonal Differencing

For seasonal data with period $s$:
$$Y'_t = Y_t - Y_{t-s}$$

```python
# For monthly data (s=12)
df['Sales_seasonal_diff'] = df['Sales'].diff(periods=12)
adf_test(df['Sales_seasonal_diff'].dropna(), 'Seasonally Differenced Sales')
```

### 3.3 Log Transform

Stabilizes variance (useful when variance increases with level):

$$Y'_t = \log(Y_t)$$

```python
df['Sales_log'] = np.log(df['Sales'])

# Then difference if still non-stationary
df['Sales_log_diff'] = df['Sales_log'].diff()
adf_test(df['Sales_log_diff'].dropna(), 'Log-Differenced Sales')
```

### 3.4 Automated Differencing

```python
def make_stationary(series, max_diff=3):
    """
    Automatically difference until stationary
    """
    d = 0
    temp = series.copy()
    
    while d <= max_diff:
        result = adfuller(temp.dropna())
        p_value = result[1]
        
        print(f"d={d}: ADF p-value = {p_value:.6f}", end='')
        
        if p_value < 0.05:
            print(" → STATIONARY ✓")
            break
        else:
            print(" → NON-STATIONARY, differencing...")
            temp = temp.diff()
            d += 1
    
    if d > max_diff:
        print(f"\n⚠️  Warning: Still non-stationary after {max_diff} differences")
    
    return temp, d

stationary_series, diff_order = make_stationary(df['Sales'])
print(f"\nOptimal differencing order: d={diff_order}")
```

---

## 4. Baselines

### 4.1 Naive Method

**Formula:** $\hat{Y}_{t+1} = Y_t$

"Tomorrow will be exactly like today"

```python
def naive_forecast(train, test):
    """
    Naive baseline: forecast = last value
    """
    last_value = train.iloc[-1]
    predictions = np.array([last_value] * len(test))
    return predictions

# Example
train_size = int(len(df) * 0.8)
train = df['Sales'][:train_size]
test = df['Sales'][train_size:]

naive_pred = naive_forecast(train, test)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test (Actual)', linewidth=2)
plt.plot(test.index, naive_pred, label='Naive Forecast', linestyle='--')
plt.legend()
plt.title('Naive Forecasting')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.2 Simple Average

**Formula:** $\hat{Y}_{t+1} = \frac{1}{t}\sum_{i=1}^{t} Y_i$

```python
def average_forecast(train, test):
    """
    Average method: forecast = mean of all historical data
    """
    mean_value = train.mean()
    return np.array([mean_value] * len(test))

avg_pred = average_forecast(train, test)
```

### 4.3 Seasonal Naive

**Formula:** $\hat{Y}_t = Y_{t-s}$

"This December will be like last December"

```python
def seasonal_naive(train, test, season_length=12):
    """
    Seasonal naive: use value from same season last year
    """
    predictions = []
    
    for i in range(len(test)):
        # Look back by season_length
        if len(train) >= season_length:
            pred = train.iloc[-(season_length - (i % season_length))]
        else:
            pred = train.iloc[-1]  # Fallback to naive
        predictions.append(pred)
    
    return np.array(predictions)

seasonal_pred = seasonal_naive(train, test, season_length=12)
```

### 4.4 Drift Method

**Formula:** $\hat{Y}_{t+h} = Y_t + h \times \frac{Y_t - Y_1}{t-1}$

Extends the line from first to last observation

```python
def drift_forecast(train, test):
    """
    Drift method: linear trend from start to end, extended
    """
    last_value = train.iloc[-1]
    first_value = train.iloc[0]
    n = len(train)
    
    # Drift (slope)
    drift = (last_value - first_value) / (n - 1)
    
    # Forecast
    predictions = [last_value + drift * (i + 1) for i in range(len(test))]
    return np.array(predictions)

drift_pred = drift_forecast(train, test)
```

### 4.5 Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(actual, predicted, method_name):
    """
    Calculate forecast accuracy metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"\n{method_name}:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Evaluate all methods
results = {}
results['Naive'] = evaluate_forecast(test, naive_pred, 'Naive')
results['Average'] = evaluate_forecast(test, avg_pred, 'Average')
results['Seasonal Naive'] = evaluate_forecast(test, seasonal_pred, 'Seasonal Naive')
results['Drift'] = evaluate_forecast(test, drift_pred, 'Drift')

# Best method
best_method = min(results.keys(), key=lambda k: results[k]['MAPE'])
print(f"\n✓ Best baseline method: {best_method}")
```

---

## 5. Exam Preparation

### 5.1 Key Formulas

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Differencing** | $Y'_t = Y_t - Y_{t-1}$ | Remove trend |
| **Log Transform** | $Y'_t = \log(Y_t)$ | Stabilize variance |
| **Seasonal Diff** | $Y'_t = Y_t - Y_{t-s}$ | Remove seasonality |
| **Naive** | $\hat{Y}_{t+1} = Y_t$ | Random walk data |
| **Seasonal Naive** | $\hat{Y}_t = Y_{t-s}$ | Strong seasonality |

### 5.2 Common Exam Questions

**Q1: ADF test gives p-value = 0.12. Is the series stationary?**

**Answer:** No, **NON-STATIONARY**
- p-value (0.12) > 0.05
- Fail to reject $H_0$ (unit root exists)
- Need to difference the series

**Q2: After first differencing, ADF p-value = 0.001. Now stationary?**

**Answer:** Yes, **STATIONARY**
- p-value (0.001) < 0.05
- Reject $H_0$
- Series is now stationary with d=1

**Q3: Given data: [100, 102, 105, 103, 108]. Calculate first difference.**

**Solution:**
$$Y'_t = Y_t - Y_{t-1}$$

- $Y'_2 = 102 - 100 = 2$
- $Y'_3 = 105 - 102 = 3$
- $Y'_4 = 103 - 105 = -2$
- $Y'_5 = 108 - 103 = 5$

**First differenced series:** [NaN, 2, 3, -2, 5]

### 5.3 Interview Questions

**Q: You difference a series twice but it's still non-stationary. What could be wrong?**

**A:** Possible issues:
1. **Strong seasonality:** Try seasonal differencing instead
2. **Non-linear trend:** Try log transform first, then difference
3. **Structural breaks:** Data has regime changes (need different approach)
4. **Outliers:** Extreme values affecting test results
5. **Too short series:** Need more data for reliable test

**Q: Your model performs worse than naive baseline. What does this tell you?**

**A:** 
- Model isn't learning useful patterns
- Data might be random walk (naive is optimal)
- Over-complicated model (Occam's razor)
- **Actions:**
  - Simplify model
  - Check if data is actually predictable
  - Try different features/transformations

---

## Summary

**Stationarity Requirements:**
1. Constant mean over time
2. Constant variance over time
3. Covariance depends only on lag, not time

**Testing:**
- **Visual:** Rolling mean/std plots
- **Statistical:** ADF test (p < 0.05 → stationary)

**Achieving Stationarity:**
- **Differencing:** Removes trend
- **Log transform:** Stabilizes variance
- **Seasonal differencing:** Removes seasonality

**Baseline Methods:**
- **Naive:** Use last value
- **Average:** Use mean
- **Seasonal Naive:** Use last season's value
- **Drift:** Linear extrapolation

**Key Insight:** Always compare advanced models to baselines!

**Next Class:** Smoothing techniques (Moving Average, Exponential Smoothing)
