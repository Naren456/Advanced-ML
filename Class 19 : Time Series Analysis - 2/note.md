# Class 19: Time Series Analysis - Smoothing Techniques

> **Core Principle:** "Removing noise to reveal underlying patterns"

---

## Table of Contents
1. [Why Smoothing?](#1-why-smoothing)
2. [Moving Average](#2-moving-average)
3. [Exponential Smoothing](#3-exponential-smoothing)
4. [Evaluation](#4-evaluation)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. Why Smoothing?

### 1.1 The Problem

**Raw time series data is noisy:**
- Random fluctuations obscure true pattern
- Difficult to identify trend
- Hard to make forecasts

**Goal:** Extract underlying signal from noise

### 1.2 Signal vs Noise

```
Noisy Data:               Smoothed Data:
Sales                     Sales
  | ●  ●                    |
  |  ●●  ●                  |     ████
  | ●  ●● ●                 |   ██    ██
  |●     ●  ●               | ██        ██
  |_________                |_____________
  Time                      Time
(Hard to see trend)       (Clear upward trend)
```

---

## 2. Moving Average

### 2.1 Simple Moving Average (SMA)

**Formula:**
$$\text{MA}_t(k) = \frac{1}{k}\sum_{i=0}^{k-1} Y_{t-i}$$

**Example:** 3-day moving average
$$\text{MA}_t(3) = \frac{Y_t + Y_{t-1} + Y_{t-2}}{3}$$

### 2.2 Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('mobilesales.xlsx', parse_dates=['DATE'], index_col='DATE')

# Calculate different window sizes
df['MA_3'] = df['Sales'].rolling(window=3).mean()
df['MA_6'] = df['Sales'].rolling(window=6).mean()
df['MA_12'] = df['Sales'].rolling(window=12).mean()

# Visualize
plt.figure(figsize=(14, 7))
plt.plot(df['Sales'], label='Original', alpha=0.5, linewidth=1)
plt.plot(df['MA_3'], label='MA-3', linewidth=2)
plt.plot(df['MA_6'], label='MA-6', linewidth=2)
plt.plot(df['MA_12'], label='MA-12', linewidth=2)
plt.legend()
plt.title('Moving Averages with Different Window Sizes')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.3 Properties

**Advantages:**
- Simple to understand and implement
- Reduces noise effectively
- No parameters except window size

**Disadvantages:**
1. **Lags behind trend:**
   - Averages past data → delayed reaction to changes
   - Larger window = more lag

2. **Equal weights:**
   - Data from 12 months ago weighted same as yesterday
   - Ignores recency

3. **Cannot forecast:**
   - Only smooths existing data
   - Need separate forecasting method

### 2.4 Choosing Window Size

**Trade-off:**
- **Small window (3-5):** Responsive but noisy
- **Large window (12+):** Smooth but lags

**Selection:**
```python
# For monthly data with yearly seasonality
window = 12  # One full cycle

# For daily data with weekly pattern
window = 7   # One full week
```

---

## 3. Exponential Smoothing

### 3.1 Simple Exponential Smoothing (SES)

**Key Idea:** Recent data more important than old data

**Formula:**
$$\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha)\hat{Y}_t$$

where $\alpha \in (0,1)$ is the smoothing parameter

**Expanded form:**
$$\hat{Y}_{t+1} = \alpha Y_t + \alpha(1-\alpha)Y_{t-1} + \alpha(1-\alpha)^2Y_{t-2} + ...$$

**Weights decay exponentially!**

### 3.2 Alpha Parameter

**Effect of $\alpha$:**

```
α = 0.9 (High):              α = 0.1 (Low):
  Weights:                     Weights:
  0.9  ●                       0.1    ●
  0.09   ●                     0.09     ●
  0.009    ●                   0.081      ●
  (Recent data dominates)      (Past data matters)
```

**When to use:**
- **High α (0.7-0.9):** Volatile data, need fast response
- **Low α (0.1-0.3):** Stable data, prefer smoothness

### 3.3 Implementation

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Fit SES model
ses_model = SimpleExpSmoothing(train['Sales'])
fitted_model = ses_model.fit(smoothing_level=0.2, optimized=False)

# Or optimize alpha automatically
fitted_model_opt = ses_model.fit()
alpha_opt = fitted_model_opt.params['smoothing_level']
print(f"Optimized alpha: {alpha_opt:.3f}")

# Forecast
forecast = fitted_model.forecast(steps=len(test))

# Plot
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test (Actual)', linewidth=2)
plt.plot(test.index, forecast, label=f'SES Forecast (α={0.2})', 
         linestyle='--', linewidth=2)
plt.legend()
plt.title('Simple Exponential Smoothing')
plt.grid(True, alpha=0.3)
plt.show()
```

### 3.4 Double Exponential Smoothing (Holt's Method)

**Adds trend component**

**Level equation:**
$$\ell_t = \alpha Y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})$$

**Trend equation:**
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$$

**Forecast:**
$$\hat{Y}_{t+h} = \ell_t + h \cdot b_t$$

**Parameters:**
- $\alpha$: Level smoothing (0-1)
- $\beta$: Trend smoothing (0-1)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt's method
holt = ExponentialSmoothing(train['Sales'], trend='add', seasonal=None)
holt_fit = holt.fit()

# Forecast
holt_forecast = holt_fit.forecast(len(test))

print(f"Alpha (level): {holt_fit.params['smoothing_level']:.3f}")
print(f"Beta (trend): {holt_fit.params['smoothing_trend']:.3f}")
```

### 3.5 Triple Exponential Smoothing (Holt-Winters)

**Adds seasonality**

**Three equations:**

**Level:**
$$\ell_t = \alpha(Y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$$

**Trend:**
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$$

**Seasonality:**
$$s_t = \gamma(Y_t - \ell_t) + (1-\gamma)s_{t-m}$$

**Forecast (Additive):**
$$\hat{Y}_{t+h} = \ell_t + h \cdot b_t + s_{t+h-m}$$

**Forecast (Multiplicative):**
$$\hat{Y}_{t+h} = (\ell_t + h \cdot b_t) \times s_{t+h-m}$$

### 3.6 Additive vs Multiplicative

**Additive ($Y = L + T + S$):**
- Seasonal amplitude constant
- Use when seasonal variation doesn't change with level

**Multiplicative ($Y = L \times T \times S$):**
- Seasonal amplitude grows with level
- Use when seasonal variation proportional to level

```python
# Additive seasonality
hw_add = ExponentialSmoothing(train['Sales'], 
                               trend='add',
                               seasonal='add',
                               seasonal_periods=12)
hw_add_fit = hw_add.fit()

# Multiplicative seasonality
hw_mul = ExponentialSmoothing(train['Sales'],
                               trend='add',
                               seasonal='mul',
                               seasonal_periods=12)
hw_mul_fit = hw_mul.fit()

# Forecast
add_forecast = hw_add_fit.forecast(len(test))
mul_forecast = hw_mul_fit.forecast(len(test))

# Compare
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(train.index, train, label='Train')
axes[0].plot(test.index, test, label='Test', linewidth=2)
axes[0].plot(test.index, add_forecast, label='Additive', linestyle='--')
axes[0].set_title('Holt-Winters Additive')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train.index, train, label='Train')
axes[1].plot(test.index, test, label='Test', linewidth=2)
axes[1].plot(test.index, mul_forecast, label='Multiplicative', linestyle='--')
axes[1].set_title('Holt-Winters Multiplicative')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. Evaluation

### 4.1 Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(actual, predicted, name='Model'):
    """
    Calculate forecast metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"\n{name}:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Evaluate all models
results = {}
results['MA-12'] = evaluate_forecast(test, ma_forecast, 'Moving Average')
results['SES'] = evaluate_forecast(test, ses_forecast, 'Simple Exp Smoothing')
results['Holt'] = evaluate_forecast(test, holt_forecast, "Holt's Method")
results['HW-Add'] = evaluate_forecast(test, add_forecast, 'HW Additive')
results['HW-Mul'] = evaluate_forecast(test, mul_forecast, 'HW Multiplicative')

# Best model
best = min(results.items(), key=lambda x: x[1]['MAPE'])
print(f"\n✓ Best Model: {best[0]} (MAPE: {best[1]['MAPE']:.2f}%)")
```

### 4.2 Residual Analysis

```python
def plot_residuals(actual, predicted, title='Residual Analysis'):
    """
    Analyze forecast residuals
    """
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(actual.index, residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(residuals, bins=20, edgecolor='black')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    
    # ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), lags=20, ax=axes[1, 0])
    axes[1, 0].set_title('ACF of Residuals')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Statistical test
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.4f} (should be ~0)")
    print(f"  Std:  {residuals.std():.4f}")

plot_residuals(test, mul_forecast, 'HW Multiplicative - Residuals')
```

---

## 5. Exam Preparation

### 5.1 Method Comparison

| Method | Components | Parameters | Best For |
|--------|------------|------------|----------|
| **SMA** | None | Window size | Quick smoothing |
| **SES** | Level | α | No trend/season |
| **Holt** | Level + Trend | α, β | Trend, no season |
| **Holt-Winters** | Level + Trend + Season | α, β, γ | Complete data |

### 5.2 Common Exam Questions

**Q1: Given sales [100, 105, 110], calculate 2-period MA.**

**Solution:**

MA at t=2: $\frac{100 + 105}{2} = 102.5$

MA at t=3: $\frac{105 + 110}{2} = 107.5$

**Result:** [NaN, 102.5, 107.5]

**Q2: With α=0.3, forecast is 100, actual is 110. Calculate next forecast.**

**Solution:**

$$\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha)\hat{Y}_t$$
$$= 0.3 \times 110 + 0.7 \times 100$$
$$= 33 + 70 = 103$$

**Q3: When to use additive vs multiplicative Holt-Winters?**

**Answer:**

**Additive:** Seasonal variation constant
```
Jan: 1000 ± 100
Jul: 2000 ± 100  (same ±100)
```

**Multiplicative:** Seasonal variation grows
```
Jan: 1000 × 1.1 = 1100
Jul: 2000 × 1.1 = 2200  (grows proportionally)
```

### 5.3 Interview Questions

**Q (E-commerce): You use MA-12 but forecasts lag 2 months behind actual trends. Fix?**

**A:** Moving Average inherently lags. Solutions:
1. **Reduce window:** Use MA-6 or MA-3 (more responsive)
2. **Switch to Exponential Smoothing:** Weights recent data more
3. **Use Holt's method:** Explicitly models trend
4. **Increase α:** If using exponential smoothing

**Q (Forecasting Role): Client wants simple, interpretable method for seasonal sales. Recommend?**

**A:** **Holt-Winters (Additive or Multiplicative)**

Reasons:
- Handles trend + seasonality
- Interpretable parameters
- Industry standard
- Easy to explain to client
- Good baseline before complex models

**Recommendation process:**
1. Plot data → check if seasonal amplitude grows
2. If constant → Additive
3. If grows → Multiplicative
4. Optimize α, β, γ automatically
5. Validate on holdout set

---

## Summary

**Key Takeaways:**
- **Moving Average:** Simple smoothing, equal weights, lags trend
- **SES:** Exponential weights, good for level-only data
- **Holt:** Adds trend component
- **Holt-Winters:** Complete model with trend + seasonality

**Parameter Meanings:**
- **α:** Level smoothing (0 = no update, 1 = just current value)
- **β:** Trend smoothing
- **γ:** Seasonal smoothing

**Model Selection:**
- No trend/season → SES
- Trend only → Holt
- Trend + Season → Holt-Winters
- Growing seasonality → Multiplicative
- Constant seasonality → Additive

**When Smoothing Fails:**
- Complex non-linear patterns → Try ARIMA
- Multiple seasonalities → Try SARIMA or ML methods
- Structural breaks → Segment data

**Next Class:** ACF/PACF and ARIMA models
