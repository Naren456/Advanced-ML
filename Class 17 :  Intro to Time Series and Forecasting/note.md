# Class 17: Introduction to Time Series Analysis

> **Core Principle:** "Forecasting is about making better decisions, not perfect predictions"

---

## Table of Contents
1. [Introduction to Time Series](#1-introduction-to-time-series)
2. [Real-World Examples](#2-real-world-examples)
3. [Case Study: Mobile Plus](#3-case-study-mobile-plus)
4. [Structure & Comparison](#4-structure--comparison)
5. [Handling Missing Values](#5-handling-missing-values)
6. [Components of Time Series](#6-components-of-time-series)
7. [Introduction to Moving Averages](#7-introduction-to-moving-averages)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction to Time Series

### 1.1 Definition
**Time Series:** Data recorded at **regular time intervals** in a **specific order/sequence**.

**Key Requirement:**
- **Sequence is Critical:** Unlike typical datasets where shuffling rows doesn't change the meaning, in Time Series, the order is the information itself. $Y_t$ is meaningless without knowing it came after $Y_{t-1}$.

### 1.2 Objectives
1.  **Finding Patterns (Analysis):** Understanding *why* things happened in the past (e.g., "Sales always drop in February").
2.  **Forecasting (Prediction):** Predicting *what* will happen in the future to enable decision-making.

---

## 2. Real-World Examples

Time series data is ubiquitous in our daily lives:

1.  **Weather Forecasting:** Temperature, rainfall, humidity recorded hourly/daily.
2.  **Stock Market:** Share prices (e.g., Tata Motors, NIFTY) changing every second.
3.  **Retail Sales:** Amazon/Flipkart daily revenue, inventory levels.
4.  **Fitness Tracking:**
    - Smartwatches recording steps per hour.
    - Daily sleep hours.
    - Weight tracking over months.
5.  **Healthcare:** Patient heart rate monitoring (ECG), blood pressure logs.
6.  **Internet Activity:** Server traffic, social media engagement rates over time.

---

## 3. Case Study: Mobile Plus

**Scenario:** "Mobile Plus" manufactures smartphones.

**The Problem:** Note the number of phones to manufacture next month.

**Risks:**
-   **Over-forecasting (Predicted > Actual demand):**
    -   Excess inventory.
    -   High warehouse costs.
    -   Capital stuck in unsold goods.
    -   Risk of obsolescence (new models coming out).
-   **Under-forecasting (Predicted < Actual demand):**
    -   Stockouts.
    -   Lost revenue.
    -   Damage to brand reputation (customers go to competitors).

**Goal:** Minimize the error between Forecast and Actuals to balance these risks.

---

## 4. Structure & Comparison

### 4.1 Structure of Time Series Data
Typically consists of just two columns:

1.  **Timestamp:**
    -   Must be **sorted** ascendingly.
    -   Frequency can be Minute, Hourly, Daily, Weekly, Quarterly, Yearly.
2.  **Value:**
    -   The metric we want to track (Stock Price, Quantity Sold, Temperature).

### 4.2 Time Series vs. ML Regression
| Feature | Standard ML Regression | Time Series Analysis |
|---------|------------------------|----------------------|
| **Independent Variables** | Requires features ($X_1, X_2...$) like Age, Salary, etc. | **Not required**. Uses its own past values. |
| **Input** | External factors | **Time itself** is the primary input. |
| **Prediction Basis** | Relationships between features | History ($Y_{t-1}, Y_{t-2}$) predicts Future ($Y_{t+1}$). |

---

## 5. Handling Missing Values

Missing data breaks the "regular interval" rule. We explored three approaches:

| Approach | Verdict | Reasoning |
|----------|---------|-----------|
| **Mean Imputation** | ❌ Not Recommended | Fills with global average, destroying trends and seasonality. |
| **Constant (e.g., 0)** | ❌ Not Recommended | Creates artificial "crashes" in data. |
| **Linear Interpolation** | ✅ **Recommended** | "Connect the dots". Takes the average of the value *just before* and *just after*. |

**Linear Interpolation Formula:**
$$Y_{missing} = \frac{Y_{before} + Y_{after}}{2}$$

*Note:* Preserves the local slope/trend of the data.

---

## 6. Components of Time Series

**Mathematical Representation:**
-   **History:** $Y_1, Y_2, Y_3, \dots, Y_t$ ($t$ = current time)
-   **Forecast:** $Y_{t+1}, Y_{t+2}, \dots$

### 6.1 Main Components
1.  **Trend:**
    -   Long-term movement in the series.
    -   Example: "Mobile Plus" sales steadily increasing over 5 years.
2.  **Seasonality:**
    -   Short-term, **repeating** patterns at fixed intervals.
    -   Example: Sales always peak in "Diwali" season or weekends.

### 6.2 Analysis of Sample Data
(Visualizing a plot)
-   **Identifying Trend:** If the graph goes from bottom-left to top-right, it's an **Uptrend**.
-   **Identifying Patterns:** Look for shapes that repeat (e.g., a "W" shape every 7 days).
-   **Domain Knowledge:** Crucial for interpretation.
    -   *Example:* A dip in sales every Tuesday might be random to a machine, but a store manager knows "Tuesday is weekly market closure".

---

## 7. Introduction to Moving Averages

### 7.1 Definition
Calculating the average of data points within a "sliding window" that moves forward with time.

### 7.2 Benefits
1.  **Reduces Noise:** Smoothing out random daily fluctuations (outliers) to see the "real" picture.
2.  **Reveals Structure:** Makes the underlying Trend much clearer.

### 7.3 Types
1.  **Centered Moving Average:** Window is centered on the current time (uses future data). Good for analysis.
2.  **Weighted Moving Average:** Assigns different importance numbers to points (typically deeper analysis - discussed in next class).

### 7.4 The "Lag" Effect
-   Moving averages introduce a **Lag**.
-   If prices drop suddenly, the Moving Average will react slowly.
-   *Benefit:* Prevents knee-jerk reactions to false alarms.
-   *Drawback:* Late entry/exit signal in trading.

---

## 8. Conclusion

1.  **Recent Data Matters:** In Time Series, $Y_t$ is usually most correlated with $Y_{t-1}$ (yesterday) rather than $Y_{t-100}$ (last year).
2.  **Domain is Key:** Machines find patterns; Humans find *meaning*.
3.  **Goal:** We don't need *perfect* predictions (impossible); we need *better* decisions than random guessing.

**Next Class:** Deep dive into implementation of Moving Averages and Time Series Decomposition.
