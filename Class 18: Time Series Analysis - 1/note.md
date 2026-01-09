# Class 18: Time Series Analysis - 1

## 1. Introduction to Time Series
**Time Series** data is a sequence of data points varying over time, recorded at specific time intervals (hourly, daily, weekly, etc.). In Time Series, the **order** of the data matters.

### Key Characteristics
*   **Sequential Data:** Order matters (unlike standard regression where input rows are independent).
*   **Time as Input:** Future values are predicted based on past values and time itself ($Y_t = f(Y_{t-1}, Y_{t-2}, \dots, t)$).
*   **Objectives:**
    1.  **Monitoring:** Finding patterns and structure (Analysis).
    2.  **Forecasting:** Predicting future values.

### Business Importance
Forecasting is crucial for efficient operations:
*   **Over-forecasting:** Leads to wasted inventory, loss of value, and stuffed stores.
*   **Under-forecasting:** Leads to lost potential revenue, unsatisfied demand, and damaged brand trust.
*   *Goal:* Forecasting is not about perfection; it's about making **better decisions**.

### Time Series vs. Regression
*   **ML Regression:** Predicts $Y$ based on features $X$ (e.g., Sales based on Price, Quantity).
*   **Time Series:** Predicts $Y$ based on its own past history (e.g., Future Sales based on Past Sales).

---

## 2. Components of Time Series
A time series signal ($Y_t$) can be decomposed into three main components:

1.  **Trend ($T_t$):** The long-term movement or direction of the data (increasing, decreasing, or stable) over a long period.
2.  **Seasonality ($S_t$):** Repeating patterns or cycles that occur at fixed intervals (e.g., sales spiking during holidays like Diwali, or weekends vs. weekdays).
3.  **Noise / Residual / Error ($E_t$):** Random fluctuations that cannot be explained by trend or seasonality.

### Decomposition Models
*   **Additive Model:** $Y_t = T_t + S_t + E_t$
    *   Use when the magnitude of seasonality remains constant regardless of the trend.
*   **Multiplicative Model:** $Y_t = T_t \times S_t \times E_t$
    *   Use when the magnitude of seasonality changes (increases/decreases) as the trend changes.

---

## 3. Data Cleaning: Handling Missing Values
Time series data often has gaps. Handling them correctly is vital to preserve the signal structure.

### Bad Approaches
*   **Mean Imputation:** Filling with the global mean. Creates unnatural flat lines and destroys local trends.
*   **Zero Imputation:** Filling with 0. Distorts the data significantly.

### Good Approach: Linear Interpolation
*   **Method:** Fills the missing value by drawing a straight line between the known point before and the known point after the gap.
*   **Advantage:** Preserves the local trend and continuity of the data.
*   **Formula:** Average of immediate neighbors (for a single missing point).

---

## 4. Smoothing Techniques: Moving Average (MA)
**Moving Average** calculates the average of data points within a sliding window.

*   **Purpose:**
    *   Reduces noise (random fluctuations).
    *   Reveals the underlying **Trend**.
*   **Window Size ($k$):**
    *   **Small $k$:** Less smoothing, captures short-term fluctuations, reacts quickly to changes.
    *   **Large $k$:** More smoothing, captures long-term trends, reacts slowly.
*   **Lag Effect:** MA is based on past data, so it "lags" behind the actual signal. Peaks and dips appear later than they actually occurred. Sudden spikes or drops are smoothed out or delayed.
*   **Weighted Moving Average:** Assigns different weights to data points (e.g., higher weight to recent data) to reduce lag.

---

## 5. Implementation Summary (`colab.ipynb`)

### 5.1 Setup and Data Loading
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
mobile_sales = pd.read_excel('mobilesales.xlsx')
mobile_sales.set_index('DATE', inplace=True) # Time series requires Date index

# Check for missing values
print(mobile_sales.isnull().sum()) 
```

### 5.2 Handling Missing Values
The notebook demonstrates why Mean Imputation is bad and Linear Interpolation is good.

```python
# BAD: Mean Imputation
# mobile_sales.Sales.fillna(mobile_sales.Sales.mean()).plot()

# GOOD: Linear Interpolation
# Fills NaNs based on neighbors
sales_imputed = mobile_sales.Sales.interpolate(method='linear')
mobile_sales['Sales'] = sales_imputed
mobile_sales['Sales'].plot()
```

### 5.3 Moving Average
Calculating Rolling Mean to smooth the data.

```python
# Rolling window of 12 (e.g., yearly trend for monthly data)
MA_12 = mobile_sales['Sales'].rolling(window=12).mean()

# Visualization
plt.plot(mobile_sales['Sales'], label='Original')
plt.plot(MA_12, label='12-Month Moving Average')
plt.legend()
plt.show()
```

### 5.4 Decomposition
Using `statsmodels` to separate Trend, Seasonality, and Noise.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive Decomposition
decomposition = seasonal_decompose(mobile_sales['Sales'], model='additive')

# Components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plotting
decomposition.plot()
plt.show()
```
