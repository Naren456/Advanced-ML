# 🎤 Real-World Interview Viva: Time Series Basics

### 🏢 Retail / Supply Chain
1.  **Q:** Differentiate between "Seasonality" and "Cyclic" patterns.
    *   **A:**
        *   **Seasonality:** Fixed, known period (e.g., Sales spike every December, Traffic spikes every 6 PM).
        *   **Cyclic:** Fluctuations with **unknown frequency**, usually due to economic factors (e.g., Stock market bull/bear runs over 2-5 years).
2.  **Q:** When would you use a Multiplicative model ($Trend 	imes Season$)?
    *   **A:** When the magnitude of the seasonality **increases** as the trend increases (e.g., Christmas sales are 10k in 2010 but 100k in 2020). If the seasonal spike is constant (+10k always), use Additive.
