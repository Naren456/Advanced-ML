# Class 20: Time Series Analysis - Advanced Smoothing and Stationarity

> **Core Principle:** "Preparing data for ARIMA modeling"

---

## Table of Contents
1. [Advanced Smoothing Techniques](#1-advanced-smoothing)
2. [Stationarity Revisited](#2-stationarity-deep-dive)
3. [Transformation Methods](#3-transformations)
4. [Model Selection](#4-selection)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. Advanced Smoothing

### 1.1 Dampened Trend

**Problem:** Linear trend forecasts can be unrealistic long-term

**Solution:** Dampen the trend over time

**Formula:**
$$\hat{Y}_{t+h} = \ell_t + (\phi + \phi^2 + ... + \phi^h)b_t$$

where $\phi \in (0,1)$ is damping parameter

**Interpretation:**
- $\phi = 1$: No damping (standard Holt)
- $\phi < 1$: Trend flattens over time
- $\phi = 0.9$: Typical value

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Dampened trend
model_dampened = ExponentialSmoothing(
    train['Sales'],
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    damped_trend=True  # Enable damping
).fit()

# Get damping parameter
phi = model_dampened.params['damping_trend']
print(f"Damping parameter φ: {phi:.3f}")

# Forecast comparison
normal_forecast = model_normal.forecast(24)
dampened_forecast = model_dampened.forecast(24)

plt.figure(figsize=(14, 7))
plt.plot(range(24), normal_forecast, label='Normal (φ=1)')
plt.plot(range(24), dampened_forecast, label=f'Dampened (φ={phi:.2f})')
plt.legend()
plt.title('Normal vs Dampened Trend Forecast')
plt.xlabel('Forecast Horizon')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.2 Grid Search for Optimal Model

```python
def grid_search_exponential_smoothing(train, test):
    """
    Find best Holt-Winters configuration
    """
    from itertools import product
    
    trends = [None, 'add', 'mul']
    seasonals = [None, 'add', 'mul']
    dampeds = [False, True]
    
    results = []
    
    for trend, seasonal, damped in product(trends, seasonals, dampeds):
        # Skip invalid combinations
        if seasonal is None and damped:
            continue
        if trend is None and damped:
            continue
        if seasonal is None and trend is None:
            continue
            
        try:
            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=12 if seasonal else None,
                damped_trend=damped
            ).fit()
            
            forecast = model.forecast(len(test))
            mape = np.mean(np.abs((test - forecast) / test)) * 100
            
            results.append({
                'trend': trend,
                'seasonal': seasonal,
                'damped': damped,
                'mape': mape,
                'aic': model.aic,
                'bic': model.bic
            })
            
        except Exception as e:
            continue
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results).sort_values('mape')
    return results_df

# Find best model
results = grid_search_exponential_smoothing(train['Sales'], test['Sales'])
print("Top 5 Models:")
print(results.head())

best = results.iloc[0]
print(f"\nBest Model:")
print(f"  Trend: {best['trend']}")
print(f"  Seasonal: {best['seasonal']}")
print(f"  Damped: {best['damped']}")
print(f"  MAPE: {best['mape']:.2f}%")
```

---

## 2. Stationarity Deep Dive

### 2.1 Types of Non-Stationarity

**Type 1: Trend Non-Stationarity**
```
Sales
  |           /
  |         /
  |       /
  |     /
  |___/___________
  Time
(Mean changes)
```
**Fix:** Differencing

**Type 2: Variance Non-Stationarity (Heteroscedasticity)**
```
Sales
  |         /\/\/\/\
  |       /\/\
  |     /\
  |___/___________
  Time
(Variance grows)
```
**Fix:** Log transform

**Type 3: Seasonal Non-Stationarity**
```
Sales
  |  /\  /\  /\
  | /  \/  \/  \
  |/____________
  Time
(Seasonal pattern)
```
**Fix:** Seasonal differencing

### 2.2 Comprehensive Stationarity Tests

```python
def comprehensive_stationarity_check(series, name='Series'):
    """
    Complete stationarity diagnostic suite
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    print(f"\n{'='*70}")
    print(f"Stationarity Analysis: {name}")
    print('='*70)
    
    # 1. ADF Test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n1. Augmented Dickey-Fuller Test")
    print(f"   H0: Series has unit root (non-stationary)")
    print(f"   Test Statistic: {adf_result[0]:.6f}")
    print(f"   p-value: {adf_result[1]:.6f}")
    print(f"   Critical Values:")
    for key, value in adf_result[4].items():
        print(f"      {key}: {value:.4f}")
    
    adf_stationary = adf_result[1] < 0.05
    print(f"   Decision: {'STATIONARY' if adf_stationary else 'NON-STATIONARY'}")
    
    # 2. KPSS Test
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"\n2. KPSS Test")
    print(f"   H0: Series is stationary")
    print(f"   Test Statistic: {kpss_result[0]:.6f}")
    print(f"   p-value: {kpss_result[1]:.6f}")
    
    kpss_stationary = kpss_result[1] > 0.05
    print(f"   Decision: {'STATIONARY' if kpss_stationary else 'NON-STATIONARY'}")
    
    # 3. Summary
    print(f"\n3. Final Conclusion:")
    if adf_stationary and kpss_stationary:
        print("   ✓ STATIONARY (both tests agree)")
        status = 'stationary'
    elif not adf_stationary and not kpss_stationary:
        print("   ✗ NON-STATIONARY (both tests agree)")
        status = 'non-stationary'
    else:
        print("   ? INCONCLUSIVE (tests disagree)")
        print("   Recommend: Visual inspection and differencing")
        status = 'unclear'
    
    return status, adf_result[1], kpss_result[1]

# Test original series
status, adf_p, kpss_p = comprehensive_stationarity_check(df['Sales'], 'Original Sales')
```

### 2.3 Automated Differencing

```python
def auto_difference(series, max_d=3, seasonal=False, m=12):
    """
    Automatically difference series until stationary
    
    Parameters:
    -----------
    series : pd.Series
        Time series to difference
    max_d : int
        Maximum differencing order
    seasonal : bool
        Apply seasonal differencing
    m : int
        Seasonal period
    """
    from statsmodels.tsa.stattools import adfuller
    
    d = 0
    temp = series.copy()
    
    print(f"Auto-differencing (max_d={max_d}):")
    print("="*60)
    
    while d <= max_d:
        # Test stationarity
        adf_p = adfuller(temp.dropna())[1]
        
        print(f"d={d}: ADF p-value = {adf_p:.6f}", end='')
        
        if adf_p < 0.05:
            print(" → STATIONARY ✓")
            break
        else:
            print(" → NON-STATIONARY, differencing...")
            if seasonal and d == 0:
                temp = temp.diff(periods=m)
                print(f"   Applied seasonal differencing (m={m})")
            else:
                temp = temp.diff()
            d += 1
    
    if d > max_d:
        print(f"\n⚠️  Warning: Still non-stationary after {max_d} differences")
        print("   Consider: log transform, detrending, or structural break analysis")
    
    return temp, d

# Auto-difference
stationary_series, diff_order = auto_difference(df['Sales'])
print(f"\nRecommended differencing order: d={diff_order}")
```

---

## 3. Transformations

### 3.1 Box-Cox Transformation

**Generalized power transform:**

$$y_{\lambda} = \begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}$$

**Special cases:**
- $\lambda = 1$: No transform
- $\lambda = 0.5$: Square root
- $\lambda = 0$: Log
- $\lambda = -1$: Inverse

```python
from scipy.stats import boxcox

# Apply Box-Cox
transformed, lambda_param = boxcox(df['Sales'])

print(f"Optimal λ: {lambda_param:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(df['Sales'])
axes[0, 0].set_title('Original Series')

axes[0, 1].hist(df['Sales'], bins=30, edgecolor='black')
axes[0, 1].set_title('Original Distribution')

# Transformed
axes[1, 0].plot(transformed)
axes[1, 0].set_title(f'Box-Cox Transformed (λ={lambda_param:.2f})')

axes[1, 1].hist(transformed, bins=30, edgecolor='black')
axes[1, 1].set_title('Transformed Distribution')

plt.tight_layout()
plt.show()

# Test stationarity
comprehensive_stationarity_check(pd.Series(transformed), 'Box-Cox Transformed')
```

### 3.2 Combined Transformations

```python
def transform_to_stationary(series):
    """
    Apply sequence of transformations to achieve stationarity
    """
    from scipy.stats import boxcox
    from statsmodels.tsa.stattools import adfuller
    
    print("Transformation Pipeline:")
    print("="*60)
    
    transformed = series.copy()
    steps = []
    
    # Step 1: Box-Cox if variance increases with level
    rolling_std = series.rolling(12).std()
    if rolling_std.corr(series.rolling(12).mean()) > 0.5:
        transformed_bc, lambda_param = boxcox(transformed)
        transformed = pd.Series(transformed_bc, index=series.index)
        steps.append(f"Box-Cox (λ={lambda_param:.3f})")
        print(f"1. Applied Box-Cox transformation")
    
    # Step 2: Seasonal differencing if seasonal
    from statsmodels.tsa.seasonal import seasonal_decompose
    try:
        decomp = seasonal_decompose(series, model='additive', period=12)
        seasonal_strength = decomp.seasonal.var() / series.var()
        if seasonal_strength > 0.1:
            transformed = transformed.diff(periods=12)
            steps.append("Seasonal Diff (m=12)")
            print(f"2. Applied seasonal differencing")
    except:
        pass
    
    # Step 3: Regular differencing until stationary
    d = 0
    while d < 2:
        adf_p = adfuller(transformed.dropna())[1]
        if adf_p < 0.05:
            break
        transformed = transformed.diff()
        d += 1
        steps.append(f"Diff (d={d})")
        print(f"3. Applied differencing (d={d})")
    
    print(f"\nFinal transformation: {' → '.join(steps)}")
    return transformed, steps

# Apply
final_series, transformation_steps = transform_to_stationary(df['Sales'])
comprehensive_stationarity_check(final_series.dropna(), 'Final Transformed')
```

---

## 4. Selection

### 4.1 Decision Tree

```
Is series stationary?
├─ Yes → Check patterns
│   ├─ No pattern → Use SES
│   ├─ Trend only → Use Holt
│   └─ Trend + Season → Use Holt-Winters
└─ No → Transform to stationary
    ├─ Log transform (if variance ↑ with level)
    ├─ Seasonal difference (if seasonal)
    ├─ Regular difference (if trend)
    └─ Use ARIMA (next class)
```

### 4.2 Model Comparison Framework

```python
def compare_all_methods(train, test):
    """
    Compare all smoothing methods
    """
    from sklearn.metrics import mean_absolute_percentage_error as mape
    
    results = {}
    
    # Baseline
    results['Naive'] = mape(test, [train.iloc[-1]] * len(test))
    
    # Moving Average
    train_full = pd.concat([train, test])
    ma_pred = train_full.rolling(12).mean().iloc[-len(test):]
    if len(ma_pred) == len(test):
        results['MA-12'] = mape(test, ma_pred)
    
    # Simple Exponential Smoothing
    ses = SimpleExpSmoothing(train).fit()
    results['SES'] = mape(test, ses.forecast(len(test)))
    
    # Holt
    holt = ExponentialSmoothing(train, trend='add').fit()
    results['Holt'] = mape(test, holt.forecast(len(test)))
    
    # Holt-Winters Additive
    hw_add = ExponentialSmoothing(train, trend='add', seasonal='add', 
                                   seasonal_periods=12).fit()
    results['HW-Add'] = mape(test, hw_add.forecast(len(test)))
    
    # Holt-Winters Multiplicative
    hw_mul = ExponentialSmoothing(train, trend='add', seasonal='mul',
                                   seasonal_periods=12).fit()
    results['HW-Mul'] = mape(test, hw_mul.forecast(len(test)))
    
    # Display results
    results_df = pd.DataFrame(list(results.items()), 
                              columns=['Method', 'MAPE'])
    results_df = results_df.sort_values('MAPE')
    
    print("\nModel Comparison (MAPE %):")
    print("="*40)
    for idx, row in results_df.iterrows():
        print(f"{row['Method']:20s}: {row['MAPE']:6.2f}%")
    
    return results_df

# Compare
comparison = compare_all_methods(train['Sales'], test['Sales'])
```

---

## 5. Exam Preparation

### 5.1 Key Concepts

**Stationarity Tests:**
- **ADF:** H0 = non-stationary (reject if p < 0.05)
- **KPSS:** H0 = stationary (reject if p < 0.05)

**Transformations:**
- **Differencing:** Removes trend
- **Log:** Stabilizes variance
- **Box-Cox:** Generalized power transform
- **Seasonal diff:** Removes seasonality

### 5.2 Common Questions

**Q1: Series has increasing variance. Which transform?**

**Answer:** **Log transform** or Box-Cox

**Reason:** Converts multiplicative to additive

Before: $Y_t = T_t \times S_t$  
After: $\log Y_t = \log T_t + \log S_t$

**Q2: ADF says stationary, KPSS says non-stationary. What to do?**

**Answer:** 
1. Visual inspection (plot rolling statistics)
2. Try differencing once
3. If still unclear, proceed with caution
4. Use domain knowledge

**Q3: After 2 differences, still non-stationary. Next steps?**

**Answer:**
1. Check for structural breaks
2. Try seasonal differencing
3. Apply log transform first
4. Consider non-linear models
5. Verify data quality

---

## Summary

**Advanced Smoothing:**
- Dampened trend for realistic long-term forecasts
- Grid search to find optimal configuration
- Combine with stationarity tests

**Achieving Stationarity:**
1. Log/Box-Cox for variance stabilization
2. Seasonal differencing for seasonality
3. Regular differencing for trend
4. Verify with ADF/KPSS tests

**Model Selection:**
- Start simple (SES)
- Add complexity as needed (Holt, HW)
- Compare multiple methods
- Validate on holdout set

**Next Steps:**
- If exponential smoothing sufficient → Done!
- If need more flexibility → ARIMA (next class)

**Next Class:** ACF, PACF, and ARIMA model identification
