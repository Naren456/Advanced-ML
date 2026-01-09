# Class 20: Time Series Analysis - 3

## 1. Introduction
This class focuses on **Smoothing Techniques** for time series forecasting. Unlike simple baselines, smoothing methods reduce noise to expose the underlying patterns (trend and seasonality) and use weighted averages of past observations for prediction.

## 2. Data Preparation (Recap)
Before modeling, data cleaning is essential.
*   **Imputation:** Simple mean imputation often fails for time series. **Linear Interpolation** is preferred as it fills gaps by averaging the valid points immediately before and after the missing value.
    ```python
    mobile_sales['Sales'] = mobile_sales['Sales'].interpolate(method='linear')
    ```

## 3. Forecasting Baselines
Simple reference models to benchmark performance.

### 3.1. Naive Approach
*   **Assumption:** The future will be exactly the same as the last observed point.
*   **Forecast:** $\hat{y}_{t+1} = y_t$
    ```python
    test_x['pred'] = train_x.Sales[-1]
    ```

### 3.2. Seasonal Naive Approach
*   **Assumption:** The future will look like the same season from the previous cycle (e.g., this January sales = last January sales).
*   **Forecast:** $\hat{y}_{t+1} = y_{t-s}$ (where $s$ is the seasonal period, e.g., 12 for monthly data).
    ```python
    for i in test_x.index:
        test_x.loc[i,'pred'] = train_x.loc[i - pd.offsets.dateOffset(year=1)]['Sales']
    ```

### 3.3. Drift Method
*   **Assumption:** The series continues to change at the average historical rate of change (drift).
*   **Forecast:** Extrapolates the trend line connecting the first and last observation.

## 4. Smoothing Techniques

### 4.1. Moving Averages (MA)
Smoothes data by averaging a fixed window of past observations.
*   **Pandas Implementation:**
    ```python
    mobile_sales['Sales'].rolling(window=3).mean()
    ```
*   **Limitation:** Gives equal weight to all old observations in the window. Lag effect is prominent.

### 4.2. Simple Exponential Smoothing (SES)
*   **Components:** Only **Level**.
*   **Concept:** Assigns exponentially decreasing weights to past observations. Recent data is weighted more heavily than older data.
*   **Parameter:** $\alpha$ (Smoothing / Level parameter), where $0 < \alpha < 1$.
    *   High $\alpha$: More responsive to recent changes (more like Naive).
    *   Low $\alpha$: Smoother, more influence from distant past.
*   **Formula:** $F_{t+1} = \alpha y_t + (1-\alpha)F_t$
*   **Code:**
    ```python
    import statsmodels.api as sm
    model = sm.tsa.SimpleExpSmoothing(mobile_sales['Sales']).fit(smoothing_level=1/(2*12))
    model.fittedvalues.plot(label='ses')
    ```
*   **Shortcoming:** Cannot handle Trend or Seasonality.

### 4.3. Double Exponential Smoothing (DES / Holt's Method)
*   **Components:** **Level + Trend**.
*   **Concept:** Adds a second equation to update the trend component over time.
*   **Parameters:**
    *   $\alpha$ (alpha): Smoothing constant for Level.
    *   $\beta$ (beta): Smoothing constant for Trend.
*   **Formulas:**
    *   Level: $L_t = \alpha y_t + (1-\alpha)(L_{t-1} + T_{t-1})$
    *   Trend: $T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$
    *   Forecast: $F_{t+h} = L_t + h \times T_t$
*   **Shortcoming:** Cannot handle Seasonality.

### 4.4. Triple Exponential Smoothing (TES / Holt-Winters' Method)
*   **Components:** **Level + Trend + Seasonality**.
*   **Concept:** The most advanced smoothing method, handling all three components.
*   **Parameters:**
    *   $\alpha$: Level
    *   $\beta$: Trend
    *   $\gamma$ (gamma): Seasonality
*   **Seasonality Type:** Can be **Additive** (constant amplitude) or **Multiplicative** (amplitude changes with level).
*   **Performance:** Generally performs best among smoothing methods for complex real-world data like sales.

## 5. Model Evaluation
Metrics used to compare forecast accuracy:
*   **MAE (Mean Absolute Error):** Average magnitude of errors.
*   **MSE (Mean Squared Error):** Penalizes large errors more heavily.
*   **MAPE (Mean Absolute Percentage Error):** errors as a percentage (scale-independent).

```python
def performane(actual, predicted):
    print("MAE:", round(mae(actual, predicted), 3))
    print("MSE:", round(mse(actual, predicted)**0.5, 3))
    print("MAPE:", round(mape(actual, predicted), 3))
```
