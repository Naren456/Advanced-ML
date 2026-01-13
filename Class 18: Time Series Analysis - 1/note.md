# Class 18: Time Series Analysis - Smoothing and Decomposition

> **Core Principle:** "Revealing the signal by smoothing the noise"

---

## Table of Contents
1. [Review of Fundamentals](#1-review-of-fundamentals)
2. [Moving Averages](#2-moving-averages)
3. [Time Series Decomposition](#3-time-series-decomposition)
4. [Practical Implementation](#4-practical-implementation)
5. [Additional Concepts](#5-additional-concepts)
6. [Exam Preparation](#6-exam-preparation)

---

## 1. Review of Fundamentals

### 1.1 Time Series Basics
**Definition:** A sequence of data points collected at regular time intervals.

**Key Characteristics:**
- **Regular Intervals:** Data must be equi-spaced (e.g., daily, monthly).
- **Order Matters:** Unlike standard regression, shuffling data destroys the information.
- **Dependency:** Current value often depends on past values ($Y_t$ depends on $Y_{t-1}$).

**Examples:**
- Weather forecasting (Temperature, Rainfall)
- Stock market trends (NIFTY, S&P 500)
- Retail sales (Amazon daily revenue)
- Fitness tracking (Daily steps, Heart rate)

**Objectives:**
1.  **Pattern Recognition:** Identifying trends and infectious cycles (Diagnosis).
2.  **Forecasting:** Predicting future values based on past patterns (Prognosis).

**Difference from ML Regression:**
| Feature | Standard ML Regression | Time Series Forecasting |
|---------|------------------------|-------------------------|
| **Input** | Multiple independent features ($X_1, X_2...$) | Time itself ($t$) and past values ($Y_{t-1}$) |
| **Assumption** | I.I.D (Independent & Identically Distributed) | High correlation between adjacent points |
| **Ordering** | Irrelevant | Crucial |

### 1.2 Handling Missing Values
Missing data in time series cannot be simply dropped or filled with global mean, as this breaks continuity.

**Imputation Methods:**
1.  **Forward Fill (Lag 1):** Propagate valid observation forward.
2.  **Backward Fill:** Use next valid observation (risky for forecasting).
3.  **Linear Interpolation (Recommended):** Connect the dots between known points.

**Linear Interpolation Formula:**
$$Y_t = Y_{t-1} + \frac{Y_{t+1} - Y_{t-1}}{2}$$
*(Simple average of neighbors for a single missing point)*

---

## 2. Moving Averages

### 2.1 Definition
A technique to smooth out short-term fluctuations (noise) and highlight longer-term trends or cycles. It calculates the average of data points within a sliding window.

**Mathematical Representation:**
For a window size $k$ (where $k=m$ in some texts):
$$\hat{Y}_t = \frac{Y_t + Y_{t-1} + \dots + Y_{t-k+1}}{k}$$
*or formally:*
$$\hat{Y}_t = \frac{1}{k} \sum_{i=0}^{k-1} Y_{t-i}$$

### 2.2 Applications
- **Trend Identification:** Filters out "jagged" daily noise to show the underlying direction.
- **Noise Reduction:** dampens the effect of outliers.
- **Visualization:** Makes graphs easier to interpret.

**Example: Mobile Sales Company**
- **Daily:** Sales fluctuate wildly (high on weekends, low on Tuesdays).
- **Moving Average (7-day):** Smooths out day-of-week effects, revealing if overall sales are growing or shrinking.

### 2.3 Types of Moving Averages

#### A. Simple Moving Average (SMA)
- Equal weight to all points in the window.
- **Pros:** Easy to calculate.
- **Cons:** Lag effect (reacts slowly to recent changes); old data has same influence as new data.

#### B. Weighted Moving Average (WMA)
- Assigns different weights ($w_i$) to data points, usually giving **more weight to recent data**.
- Formula: $\hat{Y}_t = \sum_{i=0}^{k-1} w_i Y_{t-i}$ where $\sum w_i = 1$.

### 2.4 Centered vs. Non-Centered
- **Non-Centered (Trailing):** Uses only *past* data (e.g., $t, t-1, t-2$).
    - **Use Case:** Forecasting (Real-time).
- **Centered:** Uses past *and* future data (e.g., $t-1, t, t+1$).
    - **Use Case:** Analysis/Decomposition (Historical understanding).
    - *Note:* Cannot be used for forecasting tomorrow since we don't know tomorrow's value yet.

### 2.5 Effect of Window Size (k)
- **Small $k$ (e.g., 3):** Less smoothing, follows original data closely, more noise.
- **Large $k$ (e.g., 50):** Heavy smoothing, significant lag, misses short-term turns.

---

## 3. Time Series Decomposition

### 3.1 Components of Time Series
Any time series $Y(t)$ can be broken down into:

1.  **Trend ($T_t$):** Long-term increasing or decreasing behavior.
    - Types: Uptrend, Downtrend, Changing trend.
2.  **Seasonality ($S_t$):** Patterns repeating at regular intervals (fixed period).
    - Examples: Christmas sales spikes (yearly), Movie theater traffic (weekly).
3.  **Error / Residuals ($E_t$ or $R_t$):** Random noise/irregular fluctuations not explained by Trend or Seasonality.

### 3.2 Decomposition Equations

#### A. Additive Model
Used when variations roughly stay constant in size over time.
$$Y(t) = T(t) + S(t) + E(t)$$

#### B. Multiplicative Model
Used when variations (seasonality/error) increase or decrease proportionally with the trend (e.g., sales volume grows, and so does the magnitude of Christmas spikes).
$$Y(t) = T(t) \times S(t) \times E(t)$$

### 3.3 Log Transformation
To convert a Multiplicative model into an Additive one (for easier modeling), we often take the Logarithm:
$$\log(Y_t) = \log(T_t \times S_t \times E_t)$$
$$\log(Y_t) = \log(T_t) + \log(S_t) + \log(E_t)$$

---

## 4. Practical Implementation

### 4.1 Moving Averages in Pandas
We use the `.rolling()` method.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_excel('mobilesales.xlsx', index_col='Date', parse_dates=True)

# Calculate Moving Averages
# window=7 for weekly seasonality smoothing
df['SMA_7'] = df['Sales'].rolling(window=7).mean()
df['SMA_30'] = df['Sales'].rolling(window=30).mean()

# Visualization
plt.figure(figsize=(12,6))
plt.plot(df['Sales'], label='Original', alpha=0.5)
plt.plot(df['SMA_7'], label='7-Day MA', linewidth=2)
plt.plot(df['SMA_30'], label='30-Day MA (Trend)', linewidth=2, color='black')
plt.legend()
plt.title('Sales vs Moving Averages')
plt.show()
```

### 4.2 Time Series Decomposition
We use `statsmodels` for automatic decomposition.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose
# period=12 for monthly data (or 7 for daily data with weekly pattern)
decomposition = seasonal_decompose(df['Sales'], model='additive', period=7)

# Extract Components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Visualization
plt.figure(figsize=(12,8))

plt.subplot(411)
plt.plot(df['Sales'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()
plt.show()
```

---

## 5. Additional Concepts

### 5.1 Lag in Moving Averages
**Phenomenon:** Peaks and valleys in the MA line appear *after* they occur in the real data.
**Reason:** Since MA averages past data ($t, t-1, \dots$), a sudden spike at time $t$ takes several days to significantly pull up the average. The larger the window size $k$, the greater the lag.

### 5.2 Handling Even Values of k (Centered MA)
If $k=4$ (even), the center "falls between" two time points.
**Solution:** Two-Step Process (Double Moving Average).
1.  Calculate a 4-period moving average.
2.  Calculate a 2-period average of the *result* from step 1.
This re-aligns the average exactly with the time $t$.

---

## 6. Exam Preparation

### 6.1 Key Definitions
- **Linear Interpolation:** Filling missing values by assuming a straight line between neighbors.
- **Seasonality:** Fixed, known period (e.g., every 7 days).
- **Cyclic:** Fluctuations with *unknown* or variable period (e.g., economic recessions).

### 6.2 Common Questions
**Q1: When should you use a Multiplicative model over an Additive one?**
*Answer:* When the magnitude of the seasonal fluctuations increases as the trend increases. If the seasonal peaks look "fanned out" over time, use Multiplicative.

**Q2: Why can't we use Centered Moving Averages for forecasting?**
*Answer:* A centered average for time $t$ requires data from $t+1$ (the future). In forecasting, the future is unknown, so we must use Trailing (Non-centered) averages.

**Q3: What is the impact of window size $k$ on the residuals?**
*Answer:*
- Larger $k$ -> Smooth trend -> Larger residuals (Trend captures less variance).
- Smaller $k$ -> Jagged trend -> Smaller residuals (Trend captures noise).
