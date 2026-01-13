# Class 19: Time Series Analysis - Evaluation and Baseline Forecasting

> **Core Principle:** "Establishing a baseline to measure progress"

---

## Table of Contents
1. [Announcements](#1-announcements)
2. [Recap: Decomposition](#2-recap-decomposition)
3. [Evaluation Metrics](#3-evaluation-metrics)
4. [Train-Test Split](#4-train-test-split)
5. [Baseline Forecasting Methods](#5-baseline-forecasting-methods)
6. [Introduction to Smoothing](#6-introduction-to-smoothing)

---

## 1. Announcements

### 1.1 Class Schedule
- **No Class:** Wednesday, December 24 (Holiday)
- **Next Class:** Monday, January 5

### 1.2 Grading Criteria
| Component | Weightage | Details |
|-----------|-----------|---------|
| **Project** | 30% | Submission details to be shared. Deadline: Jan 15. |
| **Viva** | 30% | Starting Jan 15. May be conducted in groups. |
| **Anthem Exam**| 40% | End of January. Online, proctored (MCQ + Subjective). |

---

## 2. Recap: Decomposition

Any time series can be broken down into three fundamental components:

1.  **Trend ($T_t$):** The long-term behavior or direction (increasing/decreasing).
2.  **Seasonality ($S_t$):** Repeating patterns at fixed intervals (e.g., every December).
3.  **Error / Residual ($E_t$):** What remains after extracting Trend and Seasonality.
    - *Formula (Additive):* $E_t = Y_t - T_t - S_t$

---

## 3. Evaluation Metrics

How do we know if our forecast is "good"? We need objective numbers.

### 3.1 MAPE (Mean Absolute Percentage Error)
$$MAPE = \frac{1}{n} \sum \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100$$

-   **Interpretation:** Represents the average error as a percentage. "On average, my forecast is off by X%".
-   **Desired Value:** Typically **< 5%** is considered excellent, but depends on industry.
-   **Edge Case:** If actual value $y_i = 0$, MAPE is **undefined** (division by zero).

### 3.2 MSE (Mean Squared Error)
$$MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$$

-   **Pros:** Penalizes large errors heavily (due to squaring).
-   **Cons:** Not in the same unit as data (e.g., "Sales squared"). Hard to interpret.

### 3.3 RMSE (Root Mean Squared Error)
$$RMSE = \sqrt{MSE}$$

-   **Pros:** Same unit as the original data ("Sales").
-   **Interpretation:** "On average, forecasts deviate by X units from actuals".

---

## 4. Train-Test Split

### 4.1 The Golden Rule of Time Series
**Never shuffle the data.**

In classical ML, we random split ($80:20$). In Time Series, this destroys the temporal dependence (leakage of future info).

### 4.2 Sequential Splitting
We must split based on **time**.

**Example:**
-   **Dataset:** 18 years of historical data.
-   **Train:** First 17 years (History for learning).
-   **Test:** Last 1 year (Future for evaluation).

*Note:* There is no separate "Y" column (Target). The "Value" column itself is both input (past) and output (future).

---

## 5. Baseline Forecasting Methods

Before building complex models (ARIMA, LSTM), we must establish a baseline. If a complex model can't beat these, it's useless.

### 5.0 Data Setup & Evaluation Function
*Extracted from Notebook:*

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, mean_absolute_percentage_error as mape

# 1. Define Split Parameters
train_len = 212
test_len = 24

# 2. Create Train/Test Sets
# Assuming 'mobile_sales' is our resampled DataFrame with a 'Sales' column
train_x = mobile_sales.iloc[:train_len].copy()
test_x = mobile_sales.iloc[train_len:].copy()

# 3. Define Evaluation Function
# Note: The notebook calculates RMSE but prints 'MSE' in the label. 
# We calculate root of MSE for the second metric.
def performane(actual, predicted):
    print("MAE:", round(mae(actual, predicted), 3))
    print("RMSE:", round(mse(actual, predicted)**0.5, 3))
    print("MAPE:", round(mape(actual, predicted), 3))
```

### 5.1 Simple Mean Forecasting
-   **Method:** Forecast = Average of *all* historical data.
-   **Result:** A horizontal (flat) line.
-   **Verdict:** **Not intelligent.** Fails to capture trend or seasonality.

```python
# Simple Mean
# Method: Forecast next values as the mean of all training values
mean_val = train_x['Sales'].mean()
test_x['pred'] = mean_val

# Evaluate
performane(test_x.Sales, test_x.pred)
```

### 5.2 Naive Approach
-   **Method:** Forecast = The *last observed value* ($Y_{t+1} = Y_t$).
-   **Result:** A horizontal (flat) line continuing the last point.
-   **Verdict:** **Not intelligent** for long horizons, but surprisingly hard to beat for stock prices (Random Walk).

```python
# Naive Approach
# Method: Forecast next values as the last observed training value
test_x['pred'] = train_x.Sales.iloc[-1]

# Evaluate & Plot
performane(test_x.Sales, test_x.pred)
test_x['Sales'].plot(style='-o', legend=True, label='Actual')
test_x['pred'].plot(style='--', legend=True, label='Naive Forecast')
```

### 5.3 Seasonal Naive Forecast
-   **Method:** Forecast = The value from the *same season last year*.
-   **Example:** Forecast for Dec 2024 = Actual Sales of Dec 2023.
-   **Pros:** Captures **Seasonality** perfectly.
-   **Cons:** Misses **Trend** (if sales are growing year-over-year, this will under-forecast).
-   **Verdict:** Uses real historic patterns. More acceptable to businesses than plain Naive.

```python
# Seasonal Naive Approach
# Method: Forecast val(t) as the value from 1 year ago (t-12 months)

for i in test_x.index:
    # Locating the date 1 year prior to index 'i'
    # Note: Corrected 'dateOffset' to 'DateOffset' from original notebook
    prev_year_date = i - pd.DateOffset(years=1)
    
    # Assigning the sales value from that past date
    test_x.loc[i, 'pred'] = train_x.loc[prev_year_date]['Sales']

# Evaluate
performane(test_x.Sales, test_x.pred)
```

### 5.4 Drift Method
Unlike the Naive method (flat line), the Drift method allows the forecast to increase or decrease over time.

**Concept:**
It draws a straight line between the **first observation** ($y_1$) and the **last observation** ($y_T$) and extends it into the future.

**Formula:**
$$\hat{y}_{T+h} = y_T + h \left( \frac{y_T - y_1}{T-1} \right)$$
*Where:*
- $h$: Forecast horizon (how many steps ahead).
- $\frac{y_T - y_1}{T-1}$: The average change (slope) over the entire history.

**Pros/Cons:**
- **Pros:** Captures the long-term **Trend** (if training data starts low and ends high, forecast goes up).
- **Cons:** Misses Seasonality; very sensitive to the first and last points (outliers can skew the slope).

**Visual:**
Think of it as drawing a "line of best fit" that is forced to connect exactly the start and end points.

```python
# Drift Method
# Method: Linear interpolation from first point to last point of training data

# 1. Get First and Last Training Values
y_t = train_x['Sales'].iloc[-1]  # Last value
y_1 = train_x['Sales'].iloc[0]   # First value
T = len(train_x)                 # Total training duration

# 2. Calculate Slope (Drift)
# Notebook formula: (y_t - y_1) / len(train)
# Standard formula often uses (T-1), but we stick to notebook logic or close approximation
m = (y_t - y_1) / T 

# 3. Generate Forecasts
# h is the step number into the future (1, 2, ..., 24)
h = np.arange(1, len(test_x) + 1)
test_x['pred'] = y_t + m * h

# Evaluate
performane(test_x.Sales, test_x.pred)
```

---

## 6. Introduction to Smoothing (Preview)

In the upcoming class, we will explore **Smoothing Techniques** to filter noise and forecast:

1.  **Moving Average Forecasting:** Non-centered MA for prediction.
2.  **Simple Exponential Smoothing (SES):** Weighted average.
3.  **Holt's Method:** Double Exponential Smoothing (Trend).
4.  **Holt-Winters Method:** Triple Exponential Smoothing (Trend + Seasonality).

*These methods progressively handle Trend, Seasonality, and Noise.*
