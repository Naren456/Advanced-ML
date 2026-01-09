# Notes: Intro to Time Series and Forecasting (Class 17)

## 1. Introduction to Time Series
*   **Definition:** Data recorded at regular time intervals where the order/sequence matters.
*   **Examples:** Stock market charts, weather forecasting, retail sales, IoT sensor data (steps, heart rate).
*   **Goal:** To find patterns over time and forecast future values to make better decisions (e.g., inventory management).
*   **Difference from Regression:**
    *   **Regression:** Uses independent variables ($X$) to predict dependent variable ($Y$). Order doesn't necessarily matter.
    *   **Time Series:** Uses past values of the variable itself (and potentially external factors) to predict future values. Time is the primary dimension ($X$ axis).

## 2. Key Components
*   **Trend:** Long-term direction of the data (increasing, decreasing, or constant).
*   **Seasonality:** Repeating patterns over a specific period (e.g., sales spiking every December).

## 3. Handling Missing Data in Time Series
Time series data often has gaps. Handling them correctly is crucial.

### Approach 1: Mean Imputation
*   Filling missing values with the overall mean of the series.
*   **Problem:** It ignores the local context, trend, and seasonality. It creates a flat line that disrupts the natural flow of the data.

### Approach 2: Linear Interpolation (Preferred for basic gaps)
*   Connects the last known point before the gap to the first known point after the gap with a straight line.
*   **Benefit:** Preserves the trend between the two points, providing a much more realistic estimate than the global mean.

## 4. Implementation Steps (Pandas)
1.  **Load Data:** Read the dataset (e.g., `mobilesales.xlsx`).
2.  **Date Conversion:** Ensure the date column is converted to datetime objects: `pd.to_datetime()`.
3.  **Set Index:** Set the date column as the index: `df.set_index('DATE', inplace=True)`.
4.  **Check Missing:** `df.isnull().sum()`.
5.  **Imputation:**
    ```python
    # Bad: Mean Imputation
    # df['Sales'].fillna(df['Sales'].mean())

    # Good: Linear Interpolation
    df['Sales_Imputed'] = df['Sales'].interpolate(method='linear')
    ```
6.  **Visualization:** Plot the original vs. imputed data to verify the fit.
