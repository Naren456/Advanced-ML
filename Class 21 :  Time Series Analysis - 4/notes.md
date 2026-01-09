# Class 21: Time Series Analysis - ACF, PACF and ARIMA Identification

> **Core Principle:** "Using correlation patterns to identify model orders"

---

## Table of Contents
1. [Autocorrelation Function (ACF)](#1-acf)
2. [Partial Autocorrelation Function (PACF)](#2-pacf)
3. [ARIMA Components](#3-arima-components)
4. [Model Identification](#4-identification)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. ACF

### 1.1 Definition

**Autocorrelation:** Correlation between $Y_t$ and $Y_{t-k}$

**Formula:**
$$\rho_k = \frac{\text{Cov}(Y_t, Y_{t-k})}{\text{Var}(Y_t)} = \frac{\sum_{t=k+1}^{n}(Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^{n}(Y_t - \bar{Y})^2}$$

**Range:** $[-1, 1]$
- $\rho_k = 1$: Perfect positive correlation
- $\rho_k = 0$: No correlation
- $\rho_k = -1$: Perfect negative correlation

### 1.2 Interpretation

**Example:**
```
Lag 1: Correlation between today and yesterday
Lag 7: Correlation between today and last week
Lag 12: Correlation between today and same month last year
```

### 1.3 Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Load data
df = pd.read_excel('mobilesales.xlsx', parse_dates=['DATE'], index_col='DATE')

# Make stationary first
df['Sales_diff'] = df['Sales'].diff().dropna()

# Plot ACF
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(df['Sales_diff'].dropna(), lags=40, ax=ax)
ax.set_title('Autocorrelation Function (ACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
plt.show()

# Get numerical values
acf_values = acf(df['Sales_diff'].dropna(), nlags=40)
print("ACF values:")
for lag, value in enumerate(acf_values[:10]):
    print(f"  Lag {lag}: {value:.3f}")
```

### 1.4 Confidence Intervals

**95% Confidence Interval:**
$$\pm \frac{1.96}{\sqrt{n}}$$

**Interpretation:**
- Spike outside confidence interval → Significant correlation
- Inside interval → Not significantly different from zero

---

## 2. PACF

### 2.1 Definition

**Partial Autocorrelation:** Correlation between $Y_t$ and $Y_{t-k}$ **after removing** effect of intermediate lags

**Difference from ACF:**
- **ACF:** Direct correlation
- **PACF:** Correlation after controlling for intervening lags

### 2.2 Example

**Scenario:** $Y_t$ depends on $Y_{t-1}$, and $Y_{t-1}$ depends on $Y_{t-2}$

**ACF:** Shows correlation at lag 1 and lag 2 (indirect)  
**PACF:** Shows correlation only at lag 1 (direct effect)

### 2.3 Implementation

```python
# Plot PACF
fig, ax = plt.subplots(figsize=(12, 6))
plot_pacf(df['Sales_diff'].dropna(), lags=40, ax=ax, method='ywm')
ax.set_title('Partial Autocorrelation Function (PACF)')
ax.set_xlabel('Lag')
ax.set_ylabel('Partial Autocorrelation')
plt.show()

# side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_acf(df['Sales_diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('ACF')

plot_pacf(df['Sales_diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('PACF')

plt.tight_layout()
plt.show()
```

---

## 3. ARIMA Components

### 3.1 AR (AutoRegressive)

**Model:** Current value depends on past **values**

**AR(p):**
$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$$

**Example AR(1):**
$$Y_t = 5 + 0.7Y_{t-1} + \epsilon_t$$

**Characteristics:**
- **ACF:** Decays gradually (exponentially or sinusoidally)
- **PACF:** Cuts off after lag $p$

### 3.2 MA (Moving Average)

**Model:** Current value depends on past **errors**

**MA(q):**
$$Y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2} + ... + \theta_q\epsilon_{t-q}$$

**Example MA(1):**
$$Y_t = 100 + \epsilon_t + 0.5\epsilon_{t-1}$$

**Characteristics:**
- **ACF:** Cuts off after lag $q$
- **PACF:** Decays gradually

### 3.3 ARMA (AutoRegressive Moving Average)

**Combines AR and MA:**

**ARMA(p,q):**
$$Y_t = c + \phi_1Y_{t-1} + ... + \phi_pY_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}$$

**Characteristics:**
- **ACF:** Decays gradually
- **PACF:** Decays gradually

### 3.4 ARIMA (Integrated ARMA)

**ARIMA(p,d,q):**
1. Difference $d$ times to make stationary
2. Apply ARMA(p,q) to differenced series

**Parameters:**
- **p:** AR order (from PACF)
- **d:** Differencing order (from stationarity tests)
- **q:** MA order (from ACF)

---

## 4. Identification

### 4.1 Pattern Recognition

**Summary Table:**

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| **AR(p)** | Decays gradually | **Cuts off** after lag p |
| **MA(q)** | **Cuts off** after lag q | Decays gradually |
| **ARMA(p,q)** | Decays gradually | Decays gradually |

### 4.2 Step-by-Step Process

```python
def identify_arima_order(series, max_lags=40):
    """
    Complete ARIMA order identification process
    """
    from statsmodels.tsa.stattools import adfuller
    
    print("ARIMA Order Identification")
    print("="*70)
    
    # Step 1: Determine d (differencing order)
    print("\nStep 1: Determine d (differencing order)")
    print("-"*70)
    
    d = 0
    temp = series.copy()
    
    while d < 3:
        adf_result = adfuller(temp.dropna())
        adf_p = adf_result[1]
        
        print(f"d={d}: ADF p-value = {adf_p:.6f}", end='')
        
        if adf_p < 0.05:
            print(" → STATIONARY ✓")
            break
        else:
            print(" → NON-STATIONARY")
            temp = temp.diff()
            d += 1
    
    print(f"\nRecommended d = {d}")
    
    # Step 2: Plot ACF and PACF
    print("\nStep 2: Analyze ACF and PACF")
    print("-"*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    plot_acf(temp.dropna(), lags=max_lags, ax=axes[0])
    axes[0].set_title(f'ACF (after d={d} differencing)')
    
    plot_pacf(temp.dropna(), lags=max_lags, ax=axes[1])
   axes[1].set_title(f'PACF (after d={d} differencing)')
    
    plt.tight_layout()
    plt.show()
    
    # Step 3: Identify p and q
    print("\nStep 3: Identify p and q from plots")
    print("-"*70)
    print("Look at the plots above:")
    print("  - ACF cuts off at lag q → MA order")
    print("  - PACF cuts off at lag p → AR order")
    print("\nSuggested starting points:")
    print("  - If ACF decays, PACF cuts at lag 2 → Try ARIMA(2,{},0)".format(d))
    print("  - If PACF decays, ACF cuts at lag 1 → Try ARIMA(0,{},1)".format(d))
    print("  - If both decay → Try ARIMA(1,{},1)".format(d))
    
    return d

# Use function
d = identify_arima_order(df['Sales'])
```

### 4.3 Automated Selection

```python
from pmdarima import auto_arima

def auto_arima_selection(series, seasonal=False, m=1):
    """
    Automatically select ARIMA order
    """
    model = auto_arima(
        series,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,  # Let it determine d
        seasonal=seasonal,
        m=m if seasonal else 1,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(f"\n{'='*70}")
    print("Best Model:", model.order)
    if seasonal:
        print("Seasonal Order:", model.seasonal_order)
    print(f"AIC: {model.aic():.2f}")
    print(f"BIC: {model.bic():.2f}")
    
    return model

# Auto-select
best_model = auto_arima_selection(df['Sales'], seasonal=True, m=12)
```

### 4.4 Manual Model Comparison

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error as mape

def compare_arima_models(train, test, orders):
    """
    Compare multiple ARIMA configurations
    """
    results = []
    
    for order in orders:
        try:
            # Fit model
            model = ARIMA(train, order=order)
            fitted = model.fit()
            
            # Forecast
            forecast = fitted.forecast(steps=len(test))
            
            # Metrics
            mape_score = mape(test, forecast)
            
            results.append({
                'order': order,
                'MAPE': mape_score,
                'AIC': fitted.aic,
                'BIC': fitted.bic
            })
            
            print(f"ARIMA{order}: MAPE={mape_score:.2%}, AIC={fitted.aic:.2f}")
            
        except Exception as e:
            print(f"ARIMA{order}: Failed - {str(e)[:50]}")
    
    # Best model
    if results:
        results_df = pd.DataFrame(results)
        best_idx = results_df['MAPE'].idxmin()
        best = results_df.loc[best_idx]
        
        print(f"\n✓ Best Model: ARIMA{best['order']}")
        print(f"  MAPE: {best['MAPE']:.2%}")
        print(f"  AIC: {best['AIC']:.2f}")
        
        return results_df
    
    return None

# Test configurations
orders_to_test = [
    (0, 1, 0),  # Random walk
    (1, 1, 0),  # AR(1) with differencing
    (0, 1, 1),  # MA(1) with differencing
    (1, 1, 1),  # ARMA(1,1) with differencing
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
]

results = compare_arima_models(train['Sales'], test['Sales'], orders_to_test)
```

---

## 5. Exam Preparation

### 5.1 Pattern Recognition Guide

**If you see this ACF/PACF pattern:**

**Pattern 1:**
- ACF: 📉 Decays gradually
- PACF: ✂️ Cuts off at lag 2
- **Model:** AR(2)

**Pattern 2:**
- ACF: ✂️ Cuts off at lag 1
- PACF: 📉 Decays gradually
- **Model:** MA(1)

**Pattern 3:**
- ACF: 📉 Decays
- PACF: 📉 Decays
- **Model:** ARMA(p,q) - try (1,1)

### 5.2 Common Exam Questions

**Q1: Given ACF cuts off at lag 2, PACF decays. Identify model.**

**Answer:** **MA(2)**

**Reason:** ACF cutoff indicates MA order

**Q2: After first differencing (d=1), PACF cuts at lag 1. Identify ARIMA order.**

**Answer:** **ARIMA(1,1,0)**
- p=1 (from PACF cutoff)
- d=1 (one differencing)
- q=0 (ACF decays)

**Q3: Compare AR(1) vs MA(1) using ACF/PACF.**

| Model | ACF | PACF |
|-------|-----|------|
| AR(1) | Decays exponentially | Cuts at lag 1 |
| MA(1) | Cuts at lag 1 | Decays exponentially |

### 5.3 Interview Questions

**Q: How to choose between multiple ARIMA models with similar performance?**

**A:** Use **principle of parsimony** (Occam's Razor):
1. Prefer simpler model (fewer parameters)
2. Compare AIC/BIC (lower is better)
3. BIC penalizes complexity more → prefer for model selection
4. Check residual diagnostics
5. Validate on holdout set

**Q: Your ACF/PACF both show significant spikes at lag 12. What does this indicate?**

**A:** **Seasonality!**
- Regular pattern at lag 12 (monthly data = yearly seasonality)
- Should use **SARIMA** instead of ARIMA
- Seasonal period: s=12

**Action:** Use SARIMA(p,d,q)(P,D,Q,12)

---

## Summary

**ACF (Autocorrelation Function):**
- Measures correlation between $Y_t$ and $Y_{t-k}$
- MA order: Count lags before cutoff
- AR: Decays gradually

**PACF (Partial Autocorrelation Function):**
- Correlation after removing intermediate effects
- AR order: Count lags before cutoff
- MA: Decays gradually

**ARIMA Identification:**
1. Test stationarity → Determine d
2. Plot ACF/PACF of differenced series
3. Identify p (PACF cutoff) and q (ACF cutoff)
4. Fit ARIMA(p,d,q)
5. Validate residuals

**Model Selection:**
- Start with auto_arima
- Try manual orders around suggested
- Compare AIC/BIC
- Prefer simpler models
- Validate on test set

**Next Class:** ARIMA implementation and SARIMA for seasonal data
