# Class 22: Time Series Analysis - ARIMA and SARIMA Models

> **Core Principle:** "Complete time series forecasting with seasonality"

---

## Table of Contents
1. [ARIMA Implementation](#1-arima-implementation)
2. [SARIMA Models](#2-sarima)
3. [Model Diagnostics](#3-diagnostics)
4. [Production Deployment](#4-production)
5. [Exam Preparation](#5-exam-preparation)

---

## 1. ARIMA Implementation

### 1.1 Building ARIMA Model

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load and prepare data
df = pd.read_excel('mobilesales.xlsx', parse_dates=['DATE'], index_col='DATE')

train_size = int(len(df) * 0.8)
train = df['Sales'][:train_size]
test = df['Sales'][train_size:]

# Fit ARIMA model
# Order determined from previous class: (p,d,q) = (1,1,1)
model = ARIMA(train, order=(1, 1, 1))
fitted_model = model.fit()

# Model summary
print(fitted_model.summary())

# Parameters
print("\nEstimated Parameters:")
print(f"AR coefficient (φ₁): {fitted_model.params['ar.L1']:.4f}")
print(f"MA coefficient (θ₁): {fitted_model.params['ma.L1']:.4f}")
```

### 1.2 Making Forecasts

```python
# Forecast
forecast_steps = len(test)
forecast = fitted_model.forecast(steps=forecast_steps)

# Get prediction intervals
forecast_result = fitted_model.get_forecast(steps=forecast_steps)
forecast_ci = forecast_result.conf_int()

# Visualize
plt.figure(figsize=(14, 7))

# Historical data
plt.plot(train.index, train, label='Train', linewidth=2)
plt.plot(test.index, test, label='Test (Actual)', linewidth=2, color='green')

# Forecast
plt.plot(test.index, forecast, label='Forecast', 
         linestyle='--', linewidth=2, color='red')

# Confidence intervals
plt.fill_between(test.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 alpha=0.3, color='red',
                 label='95% Confidence Interval')

plt.legend(loc='best')
plt.title('ARIMA(1,1,1) Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.3 Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(actual, predicted, model_name='Model'):
    """
    Comprehensive forecast evaluation
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Directional accuracy
    actual_direction = np.sign(actual.diff().dropna())
    pred_direction = np.sign(pd.Series(predicted).diff().dropna())
    directional_accuracy = (actual_direction == pred_direction).sum() / len(actual_direction) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"{'='*50}")
    print(f"MAE                    : {mae:.2f}")
    print(f"RMSE                   : {rmse:.2f}")
    print(f"MAPE                   : {mape:.2f}%")
    print(f"Directional Accuracy   : {directional_accuracy:.2f}%")
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 
            'Direction': directional_accuracy}

# Evaluate
metrics = evaluate_forecast(test, forecast, 'ARIMA(1,1,1)')
```

---

## 2. SARIMA

### 2.1 Understanding Seasonal ARIMA

**SARIMA(p,d,q)(P,D,Q)s**

**Non-seasonal part:**
- p: AR order
- d: Differencing order
- q: MA order

**Seasonal part:**
- P: Seasonal AR order
- D: Seasonal differencing order
- Q: Seasonal MA order
- s: Seasonal period (12 for monthly, 4 for quarterly)

**Full Model:**
$$\Phi_P(B^s)\phi(B)\nabla^D_s\nabla^d Y_t = \Theta_Q(B^s)\theta(B)\epsilon_t$$

### 2.2 Implementation

```python
# SARIMA model
# Example: SARIMA(1,1,1)(1,1,1,12)
sarima_model = SARIMAX(train,
                        order=(1, 1, 1),           # Non-seasonal
                        seasonal_order=(1, 1, 1, 12))  # Seasonal, period=12

sarima_fitted = sarima_model.fit(disp=False)

# Summary
print(sarima_fitted.summary())

# Forecast
sarima_forecast = sarima_fitted.forecast(steps=len(test))

# Plot
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', linewidth=2)
plt.plot(test.index, forecast, label='ARIMA', linestyle='--', alpha=0.7)
plt.plot(test.index, sarima_forecast, label='SARIMA', linestyle='--', linewidth=2)
plt.legend()
plt.title('ARIMA vs SARIMA Comparison')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.3 Auto SARIMA

```python
from pmdarima import auto_arima

# Automatic SARIMA selection
auto_model = auto_arima(
    train,
    start_p=0, start_q=0, max_p=3, max_q=3,
    seasonal=True,
    m=12,  # Monthly seasonality
    start_P=0, start_Q=0, max_P=2, max_Q=2,
    d=None, D=None,  # Auto-determine differencing
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("\n" + "="*70)
print("Best Model Selected by Auto ARIMA:")
print("="*70)
print(f"Order: {auto_model.order}")
print(f"Seasonal Order: {auto_model.seasonal_order}")
print(f"AIC: {auto_model.aic():.2f}")
print(f"BIC: {auto_model.bic():.2f}")

# Forecast
auto_forecast = auto_model.predict(n_periods=len(test))

# Evaluate
evaluate_forecast(test, auto_forecast, f'Auto-SARIMA{auto_model.order}×{auto_model.seasonal_order}')
```

---

## 3. Diagnostics

### 3.1 Residual Analysis

```python
def comprehensive_residual_diagnostics(model, name='Model'):
    """
    Complete residual diagnostic suite
    """
    residuals = model.resid
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{name} - Residual Diagnostics', fontsize=14, y=1.02)
    
    # 1. Residuals over time
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_ylabel('Residual')
    axes[0,  0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', density=True)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].set_xlabel('Residual')
    
    # Add normal curve
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * 
                    np.exp(-(x - mu)**2 / (2 * sigma**2)),
                    'r-', linewidth=2, label='Normal')
    axes[0, 1].legend()
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot')
    
    # 4. ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=20, ax=axes[1, 0])
    axes[1, 0].set_title('ACF of Residuals')
    
    # 5. PACF of residuals
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(residuals, lags=20, ax=axes[1, 1])
    axes[1, 1].set_title('PACF of Residuals')
    
    # 6. Residuals squared (heteroscedasticity check)
    axes[1, 2].plot(residuals**2)
    axes[1, 2].set_title('Squared Residuals (Check Heteroscedasticity)')
    axes[1, 2].set_ylabel('Residual²')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print(f"\n{name} - Statistical Tests:")
    print("="*70)
    
    # Ljung-Box test (residuals should be uncorrelated)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    print("\n1. Ljung-Box Test (H0: No autocorrelation in residuals)")
    print(lb_test)
    
    if (lb_test['lb_pvalue'] > 0.05).all():
        print("   ✓ Residuals appear to be white noise (good!)")
    else:
        print("   ✗ Some autocorrelation remains (model may be inadequate)")
    
    # Jarque-Bera test (normality)
    from scipy.stats import jarque_bera
    jb_stat, jb_pvalue = jarque_bera(residuals)
    print(f"\n2. Jarque-Bera Test (H0: Residuals are normally distributed)")
    print(f"   Test Statistic: {jb_stat:.4f}")
    print(f"   p-value: {jb_pvalue:.4f}")
    
    if jb_pvalue > 0.05:
        print("   ✓ Residuals are approximately normal (good!)")
    else:
        print("   ✗ Residuals deviate from normality")
    
    # Durbin-Watson (autocorrelation at lag 1)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuals)
    print(f"\n3. Durbin-Watson Statistic: {dw:.4f}")
    print("   (Should be close to 2; <1 or >3 indicates autocorrelation)")
    
    if 1.5 < dw < 2.5:
        print("   ✓ No significant autocorrelation at lag 1")
    else:
        print("   ✗ Autocorrelation detected")

# Run diagnostics
comprehensive_residual_diagnostics(sarima_fitted, 'SARIMA(1,1,1)(1,1,1,12)')
```

### 3.2 Model Comparison

```python
def compare_all_models(train, test):
    """
    Compare ARIMA, SARIMA, and baselines
    """
    from sklearn.metrics import mean_absolute_percentage_error as mape
    
    results = []
    
    # Baseline: Naive
    naive_pred = [train.iloc[-1]] * len(test)
    results.append({
        'Model': 'Naive',
        'MAPE': mape(test, naive_pred),
        'AIC': np.nan,
        'BIC': np.nan
    })
    
    # Baseline: Seasonal Naive
    seasonal_naive_pred = []
    for i in range(len(test)):
        idx = -(12 - (i % 12))
        seasonal_naive_pred.append(train.iloc[idx])
    results.append({
        'Model': 'Seasonal Naive',
        'MAPE': mape(test, seasonal_naive_pred),
        'AIC': np.nan,
        'BIC': np.nan
    })
    
    # Exponential Smoothing
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hw = ExponentialSmoothing(train, trend='add', seasonal='mul', 
                               seasonal_periods=12).fit()
    hw_pred = hw.forecast(len(test))
    results.append({
        'Model': 'Holt-Winters',
        'MAPE': mape(test, hw_pred),
        'AIC': hw.aic,
        'BIC': hw.bic
    })
    
    # ARIMA
    arima = ARIMA(train, order=(1,1,1)).fit()
    arima_pred = arima.forecast(len(test))
    results.append({
        'Model': 'ARIMA(1,1,1)',
        'MAPE': mape(test, arima_pred),
        'AIC': arima.aic,
        'BIC': arima.bic
    })
    
    # SARIMA
    sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    sarima_pred = sarima.forecast(len(test))
    results.append({
        'Model': 'SARIMA(1,1,1)(1,1,1,12)',
        'MAPE': mape(test, sarima_pred),
        'AIC': sarima.aic,
        'BIC': sarima.bic
    })
    
    # Display results
    results_df = pd.DataFrame(results).sort_values('MAPE')
    
    print("\nModel Comparison:")
    print("="*70)
    print(results_df.to_string(index=False))
    
    print(f"\n✓ Best Model (by MAPE): {results_df.iloc[0]['Model']}")
    
    return results_df

# Compare all
comparison_results = compare_all_models(train, test)
```

---

## 4. Production

### 4.1 Production-Ready Forecaster

```python
class TimeSeriesForecaster:
    """
    Production-ready time series forecasting system
    """
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.training_data = None
        
    def fit(self, data):
        """
        Train the model
        """
        self.training_data = data
        try:
            self.model = SARIMAX(data,
                                order=self.order,
                                seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit(disp=False)
            
            print(f"✓ Model fitted successfully")
            print(f"  AIC: {self.fitted_model.aic:.2f}")
            print(f"  BIC: {self.fitted_model.bic:.2f}")
            
            return self
            
        except Exception as e:
            print(f"✗ Model fitting failed: {e}")
            return None
    
    def forecast(self, steps, return_conf_int=False):
        """
        Generate forecast
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if return_conf_int:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            return forecast, conf_int
        else:
            return self.fitted_model.forecast(steps=steps)
    
    def update(self, new_data):
        """
        Update model with new data (refit)
        """
        updated_data = pd.concat([self.training_data, new_data])
        return self.fit(updated_data)
    
    def plot_forecast(self, steps=12, historical_points=36):
        """
        Visualize forecast with confidence intervals
        """
        forecast, conf_int = self.forecast(steps, return_conf_int=True)
        
        # Generate forecast index
        last_date = self.training_data.index[-1]
        freq = pd.infer_freq(self.training_data.index)
        forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Historical data (last N points)
        hist_data = self.training_data[-historical_points:]
        ax.plot(hist_data.index, hist_data, label='Historical', linewidth=2)
        
        # Forecast
        ax.plot(forecast_index, forecast, 
                label='Forecast', color='red', linestyle='--', linewidth=2)
        
        # Confidence interval
        ax.fill_between(forecast_index,
                       conf_int.iloc[:, 0],
                       conf_int.iloc[:, 1],
                       alpha=0.3, color='red',
                       label='95% Confidence Interval')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(f'Forecast: SARIMA{self.order}×{self.seasonal_order}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath):
        """
        Save model to disk
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from disk
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return model

# Usage
forecaster = TimeSeriesForecaster(order=(1,1,1), seasonal_order=(1,1,1,12))
forecaster.fit(train)

# Forecast
forecaster.plot_forecast(steps=24)

# Save for deployment
forecaster.save('sales_forecaster.pkl')

# Load and use
loaded_forecaster = TimeSeriesForecaster.load('sales_forecaster.pkl')
future_forecast = loaded_forecaster.forecast(steps=12)
print("\n12-Month Forecast:")
print(future_forecast)
```

---

## 5. Exam Preparation

### 5.1 Complete ARIMA Workflow

```
1. Data Preparation
   ├─ Handle missing values (interpolate)
   ├─ Check for outliers
   └─ Train-test split (temporal)

2. Stationarity
   ├─ ADF/KPSS tests
   ├─ Transform if needed (log, Box-Cox)
   └─ Difference to achieve stationarity (d)

3. Identify Orders
   ├─ Plot ACF/PACF
   ├─ Identify p (PACF cutoff)
   ├─ Identify q (ACF cutoff)
   └─ Check for seasonality (seasonal lags)

4. Fit Model
   ├─ ARIMA(p,d,q) or SARIMA(p,d,q)(P,D,Q,s)
   ├─ Check AIC/BIC
   └─ Compare multiple models

5. Diagnostics
   ├─ Residual analysis (should be white noise)
   ├─ Ljung-Box test (p > 0.05)
   ├─ Check normality
   └─ ACF/PACF of residuals

6. Forecast
   ├─ Generate predictions
   ├─ Confidence intervals
   └─ Evaluate on test set

7. Deploy
   ├─ Monitor performance
   ├─ Retrain periodically
   └─ Update with new data
```

### 5.2 Key Formulas

**ARIMA(p,d,q):**
$$\phi(B)(1-B)^d Y_t = \theta(B)\epsilon_t$$

where:
- $\phi(B) = 1 - \phi_1B - ... - \phi_pB^p$ (AR polynomial)
- $\theta(B) = 1 + \theta_1B + ... + \theta_qB^q$ (MA polynomial)
- $B$ is backshift operator: $BY_t = Y_{t-1}$

**SARIMA adds:**
$$\Phi(B^s)$$ (seasonal AR) and $$\Theta(B^s)$$ (seasonal MA)

### 5.3 Common Exam Questions

**Q1: You fit ARIMA(2,1,1) and residuals show significant ACF at lag 12. What's wrong?**

**Answer:** **Missing seasonality!**
- Lag 12 spike indicates yearly seasonality (monthly data)
- Should use SARIMA(2,1,1)(P,D,Q,12)
- Start with SARIMA(2,1,1)(1,0,1,12)

**Q2: Model A: AIC=500, BIC=520. Model B: AIC=510, BIC=515. Which to choose?**

**Answer:** **Model B**
- Lower BIC (515 < 520)
- BIC more reliable for model selection (penalizes complexity more)
- Slightly higher AIC acceptable

**Q3: ARIMA forecast confidence intervals widen as horizon increases. Why?**

**Answer:**
- Uncertainty compounds over time
- Each step depends on previous (uncertain) forecast
- Mathematically: Forecast variance increases with h
- Long-term forecasts less reliable
- Normal behavior, not a problem

### 5.4 Interview Questions

**Q (Data Scientist): Your SARIMA model works great on validation but fails in production. What happened?**

**A:** Possible causes:
1. **Concept drift:** Data distribution changed
2. **Seasonality changed:** Pattern shifted
3. **Structural break:** Market disruption, pandemic, etc.
4. **Data quality issues:** Production data different from training
5. **Not retrained:** Model static while world changes

**Solutions:**
- Monitor forecast errors continuously
- Retrain regularly (monthly/quarterly)
- Implement alerts for degraded performance
- A/B test new models before deployment
- Use ensemble with multiple models

**Q (ML Engineer): How to speed up SARIMA training on large dataset?**

**A:** Optimization strategies:
1. **Sampling:** Train on recent subset (last 3-5 years)
2. **Decimation:** Aggregate to lower frequency (daily→weekly)
3. **Parallel search:** Use `n_jobs=-1` in auto_arima
4. **Limit search space:** Constrain max p, q, P, Q
5. **Incremental updates:** Don't refit from scratch, use `append()`
6. **Cache results:** Save fitted models, update only when needed

---

## Summary

**ARIMA Recap:**
- **AR(p):** Past values predict current
- **I(d):** Differencing for stationarity
- **MA(q):** Past errors predict current

**SARIMA Extension:**
- Adds seasonal AR, I, MA components
- Format: (p,d,q)(P,D,Q,s)
- Essential for seasonal data

**Model Building:**
1. Make stationary
2. Identify orders (ACF/PACF)
3. Fit and validate
4. Check residuals
5. Fore cast

**Diagnostics:**
- Residuals should be white noise
- Ljung-Box p > 0.05
- ACF/PACF within confidence bounds
- Normally distributed residuals

**Production:**
- Save/load models
- Monitor performance
- Retrain periodically
- Version control models
- A/B test changes

**Congratulations! You've completed the Time Series Analysis sequence!** 🎉

Next steps: Explore Facebook Prophet, LSTM, or other advanced methods for complex patterns!
