# Class 17: Introduction to Time Series and Forecasting

> **Core Principle:** "Understanding temporal data patterns and dependencies"

---

## Table of Contents
1. [What is Time Series?](#1-what-is-time-series)
2. [Time Series Components](#2-components)
3. [Decomposition](#3-decomposition)
4. [Data Preparation](#4-data-preparation)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. What is Time Series?

### 1.1 Definition

**Time Series:** Sequence of observations recorded at successive time points

**Key Characteristics:**
- **Temporal ordering:** Order matters (cannot shuffle)
- **Dependencies:** $Y_t$ depends on $Y_{t-1}, Y_{t-2}, ...$
- **Indexed by time:** Regular intervals (hourly, daily, monthly)

### 1.2 Examples

| Domain | Time Series | Frequency |
|--------|-------------|-----------|
| **Finance** | Stock prices | Minute/Daily |
| **Retail** | Product sales | Daily/Monthly |
| **Weather** | Temperature | Hourly |
| **Healthcare** | Heart rate | Second |
| **Energy** | Power consumption | 15-min |

### 1.3 vs Standard Machine Learning

**Standard ML:**
```
Input: [Age, Income, Education] → Output: [Loan Approved]
Assumption: Rows are independent
```

**Time Series:**
```
Input: [Sales_t-1, Sales_t-2, ...] → Output: [Sales_t]
Dependency: Current value depends on past values
```

---

## 2. Components

### 2.1 Four Main Components

**1. Trend (T)**
- Long-term direction
- Increasing, decreasing, or constant
- Example: Global temperature rising

**2. Seasonality (S)**
- Regular, periodic fluctuations
- Fixed period (weekly, monthly, yearly)
- Example: Ice cream sales peak in summer

**3. Cyclical (C)**
- Longer fluctuations without fixed period
- Usually > 1 year
- Example: Economic boom/bust cycles

**4. Residual/Noise (R)**
- Random variations
- What remains after removing T, S, C

### 2.2 Decomposition Models

**Additive Model:**
$$Y_t = T_t + S_t + C_t + R_t$$

**When to use:** Seasonal variations constant in magnitude

**Example:**
```
Jan sales: 1000 + 200 (season) = 1200
Jul sales: 1000 + 200 (season) = 1200
(Same absolute seasonal effect regardless of level)
```

**Multiplicative Model:**
$$Y_t = T_t \times S_t \times C_t \times R_t$$

**When to use:** Seasonal variations proportional to level

**Example:**
```
Jan sales: 1000 × 1.2 = 1200
Jul sales: 2000 × 1.2 = 2400
(Seasonal effect grows with trend)
```

### 2.3 Visual Identification

```
Additive:                 Multiplicative:
Sales                     Sales
  |  /\/\/\/\/\            |    /\/\/\/\
  | /          \           |   /        \
  |/            \          |  /          \
  |_____________          |_/____________
  Time                     Time
(Constant amplitude)     (Growing amplitude)
```

---

## 3. Decomposition

### 3.1 Classical Decomposition

**Steps for Additive:**

1. **Compute trend** using moving average:
   $$T_t = \text{MA}_m(Y_t)$$
   
2. **Detrend:** $Y_t - T_t$

3. **Compute seasonal component:**
   Average detrended values for each season

4. **Residual:** $R_t = Y_t - T_t - S_t$

### 3.2 Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
df = pd.read_excel('mobilesales.xlsx', parse_dates=['DATE'], index_col='DATE')

# Perform decomposition
decomposition = seasonal_decompose(df['Sales'], 
                                   model='additive',  # or 'multiplicative'
                                   period=12)         # 12 for monthly data

# Extract components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(df['Sales'])
axes[0].set_ylabel('Original')
axes[0].set_title('Time Series Decomposition')

axes[1].plot(trend)
axes[1].set_ylabel('Trend')

axes[2].plot(seasonal)
axes[2].set_ylabel('Seasonality')

axes[3].plot(residual)
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')

plt.tight_layout()
plt.show()

# Verify decomposition
reconstructed = trend + seasonal + residual
print("Reconstruction error:", np.nansum((df['Sales'] - reconstructed)**2))
```

### 3.3 STL Decomposition (Better Method)

**STL = Seasonal and Trend decomposition using Loess**

**Advantages:**
- Handles any seasonality (not just monthly)
- Robust to outliers
- Seasonal component can change over time

```python
from statsmodels.tsa.seasonal import STL

stl = STL(df['Sales'], seasonal=13)  # seasonal window
result = stl.fit()

fig = result.plot()
plt.show()
```

---

## 4. Data Preparation

### 4.1 Handling Missing Values

**❌ Wrong Approach:**
```python
# DON'T DO THIS - destroys temporal structure
df['Sales'].fillna(df['Sales'].mean())
```

**✓ Correct Approach:**
```python
# Linear interpolation preserves trend
df['Sales'] = df['Sales'].interpolate(method='linear')

# For more complex patterns
df['Sales'] = df['Sales'].interpolate(method='time')  # Time-weighted
df['Sales'] = df['Sales'].interpolate(method='polynomial', order=2)
```

**Visualization:**
```python
# Create artificial gaps
df_missing = df.copy()
df_missing.loc['2020-03':'2020-05', 'Sales'] = np.nan

# Interpolate
df_filled = df_missing.interpolate(method='linear')

# Compare
plt.figure(figsize=(12, 6))
plt.plot(df['Sales'], label='Original', linewidth=2)
plt.plot(df_filled['Sales'], label='Interpolated', linestyle='--')
plt.legend()
plt.title('Missing Value Imputation')
plt.show()
```

### 4.2 Train-Test Split

**Critical Rule:** NEVER shuffle! Maintain temporal order

**✓ Correct:**
```python
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

print(f"Train: {train.index[0]} to {train.index[-1]}")
print(f"Test: {test.index[0]} to {test.index[-1]}")
```

**Time-based split example:**
```python
# Split by date
train = df[df.index < '2022-01-01']
test = df[df.index >= '2022-01-01']
```

### 4.3 Feature Engineering

**Time-based features:**
```python
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
```

**Lag features:**
```python
df['lag_1'] = df['Sales'].shift(1)   # Yesterday
df['lag_7'] = df['Sales'].shift(7)   # Last week
df['lag_30'] = df['Sales'].shift(30) # Last month
```

**Rolling statistics:**
```python
df['rolling_mean_7'] = df['Sales'].rolling(window=7).mean()
df['rolling_std_7'] = df['Sales'].rolling(window=7).std()
df['rolling_min_7'] = df['Sales'].rolling(window=7).min()
df['rolling_max_7'] = df['Sales'].rolling(window=7).max()
```

---

## 5. Exam Preparation

### 5.1 Key Concepts

**Decomposition:**
- Additive: $Y = T + S + R$
- Multiplicative: $Y = T \times S \times R$

**When to use each:**
- **Additive:** Constant seasonal amplitude
- **Multiplicative:** Growing seasonal amplitude

### 5.2 Common Exam Questions

**Q1: Explain the difference between seasonality and cyclical patterns.**

**Answer:**

| Aspect | Seasonality | Cyclical |
|--------|-------------|----------|
| **Period** | Fixed (e.g., 12 months) | Variable (2-10 years) |
| **Duration** | ≤ 1 year | > 1 year |
| **Predictability** | High | Low |
| **Cause** | Calendar effects | Economic conditions |

**Example:**
- **Seasonal:** Retail sales spike every December (fixed)
- **Cyclical:** Housing market boom/bust (irregular timing)

**Q2: Given sales data: Jan=100, Jul=110, Next Jan=150, Next Jul=180. Is this additive or multiplicative seasonality?**

**Solution:**

**Test Additive:**
- Seasonal effect Year 1: $110 - 100 = 10$
- Seasonal effect Year 2: $180 - 150 = 30$
- Different! ❌ Not additive

**Test Multiplicative:**
- Seasonal factor Year 1: $110/100 = 1.1$
- Seasonal factor Year 2: $180/150 = 1.2$
- Not exactly same but closer. Could be multiplicative with trend

**Answer:** Likely **multiplicative** (seasonal amplitude grows with level)

**Q3: Why can't we use standard train-test splitting (random shuffle) for time series?**

**Answer:**
1. **Destroys temporal dependencies:** Future data leaks into past
2. **Unrealistic:** Can't use future to predict past
3. **Evaluation bias:** Overestimates performance

**Example of failure:**
```
Shuffled:
Train: [Jan, Mar, May, Jul, Sep, Nov]
Test:  [Feb, Apr, Jun, Aug, Oct, Dec]

Problem: Training on Nov to predict Feb!
```

### 5.3 Interview Questions

**Q (Tech Company): You decompose a time series and find trend is stronger than seasonality. What does this tell you about the data?**

**A:** 
- **Long-term growth/decline dominates** short-term fluctuations
- **Business interpretation:** Sustained growth (or decline) trend
- **Forecasting approach:** Focus on trend models (Holt's method), seasonality is secondary
- **Risk:** Trend may not continue indefinitely (check for saturation)

**Q (Retail): Your decomposition shows increasing seasonal amplitude over time. Which model should you use?**

**A:** **Multiplicative model**
- Seasonal variations proportional to level
- As baseline sales grow, seasonal spikes grow too
- Formula: $Y_t = T_t \times S_t \times R_t$
- Alternative: Log-transform then use additive: $\log Y_t = T_t + S_t + R_t$

---

## Summary

**Key Takeaways:**
1. Time series has temporal ordering - never shuffle
2. Four components: Trend, Seasonality, Cyclical, Residual
3. Additive vs multiplicative based on seasonal amplitude
4. Decomposition reveals underlying patterns
5. Proper data preparation critical (interpolation, not mean imputation)

**Components Summary:**
- **Trend:** Long-term direction
- **Seasonality:** Fixed periodic patterns
- **Cyclical:** Long irregular fluctuations
- **Residual:** Random noise

**Decomposition Models:**
- **Additive:** Use when seasonal variation constant
- **Multiplicative:** Use when seasonal variation grows with level

**Next Class:** Stationarity and baseline forecasting methods
