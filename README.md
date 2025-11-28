# Geometric Brownian Motion on the Nikkei 225

*Estimating Drift & Volatility from Historical Data*

This repository analyzes the Nikkei 225 index using a **Geometric Brownian Motion (GBM)** model.
We estimate the **daily drift (α)** and **volatility (σ)** from historical data and visualize:

* The actual Nikkei 225 index
* The GBM exponential trend
* The distribution of log-returns and its normal approximation

---

## 1. What is Geometric Brownian Motion?

GBM is defined as:

[
dS_t = \alpha S_t , dt + \sigma S_t , dW_t
]

Where:

| Symbol     | Meaning                 |
| ---------- | ----------------------- |
| ( S_t )    | Price at time t         |
| ( \alpha ) | Drift (expected return) |
| ( \sigma ) | Volatility              |
| ( dW_t )   | Brownian motion         |

Log returns follow:

[
\log\left(\frac{S_{t+\Delta t}}{S_t}\right)
\sim \mathcal{N}((\alpha - \tfrac{1}{2}\sigma^2)\Delta t, , \sigma^2 \Delta t)
]

---

## 2. Parameter Estimation (α and σ)

Using daily historical Nikkei 225 price data:

**Estimated parameters:**

* **Daily drift:**
  [
  \alpha = 0.00044
  ]
* **Daily volatility:**
  [
  \sigma = 0.01223
  ]

Interpretation:

* The index grows **+0.044% per day** on average
* The daily standard deviation of returns is **1.22%**

---

## 3. Visualization

### 3.1 Nikkei 225 vs GBM exponential trend

We compare:

* Actual Nikkei 225
* Exponential fit ( S(0)e^{\alpha t} )

### 3.2 Log-return distribution

The histogram of log-returns is compared to a normal distribution:

[
\mathcal{N}(\mu, \sigma^2)
]

*(Insert your output plot here)*

---

## 4. Code Used for Estimation & Plotting

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

# Load Nikkei index data
df = pd.read_csv("nikkei_stock_average_daily_jp.csv", encoding="cp932")
df['Date'] = pd.to_datetime(df['データ日付'], format="%Y/%m/%d")

# Compute elapsed days
start_date = df['Date'].iloc[0]
df['ElapsedDays'] = (df['Date'] - start_date).dt.days
elapsed_days = df['ElapsedDays'].to_numpy()
end_values = df['終値'].to_numpy()

# Log-returns
log_returns = np.diff(np.log(end_values))
delta_t = np.diff(elapsed_days)
n = len(delta_t)

# MLE estimation
mu_hat = np.sum(log_returns) / np.sum(delta_t)
sigma_hat = np.sqrt(np.sum((log_returns - mu_hat * delta_t)**2 / delta_t) / n)
alpha_hat = mu_hat + 0.5 * sigma_hat**2

print(f"Daily drift α : {alpha_hat:.5f}")
print(f"Daily volatility σ : {sigma_hat:.5f}")

# Plotting
fig = plt.figure(figsize=(15, 6))

# (Left) Nikkei vs exponential fit
ax1 = fig.add_subplot(121)
ax1.plot(elapsed_days, end_values, label=r'$S(t)$')
ax1.plot(elapsed_days, end_values[0] * np.exp(alpha_hat*elapsed_days),
         label=rf'$S(0)e^{{\alpha t}}, \alpha={alpha_hat:.5f}$')
ax1.set_title("Nikkei 225 vs exponential fitting")
ax1.set_xlabel("Elapsed days since " + str(start_date.date()))
ax1.set_ylabel("Nikkei 225 level")
ax1.legend()
ax1.grid(True)

# (Right) log returns distribution
ax2 = fig.add_subplot(122)
plt.hist(log_returns, bins=30, density=True, alpha=0.6, color='skyblue')
x = np.linspace(min(log_returns), max(log_returns), 1000)
pdf = stats.norm.pdf(x, loc=mu_hat, scale=sigma_hat * np.sqrt(delta_t.mean()))
plt.plot(x, pdf, "r-", lw=2)
ax2.set_title(rf"Log returns and normal fit ($\sigma$={sigma_hat:.5f})")
ax2.set_xlabel("log returns")
ax2.set_ylabel("density")
ax2.grid(True)

plt.show()
```

---

## 5. Key Insights

* GBM gives a **reasonable long-term approximation** of Nikkei’s trend.
* The log-return histogram is close to normal — consistent with GBM assumptions.
* Deviations from normality (fat tails, volatility clustering) suggest that
  **Heston models or jump-diffusion** may capture real markets better.

---

## 6. Future Extensions

* Fit a **Heston stochastic volatility model**
* Implement **jump diffusion (Merton model)**
* Use α and σ to **simulate future Nikkei paths**
* Use σ to price options via **Black–Scholes**

---

## References

【金融工学】日経平均株価に対する幾何ブラウン運動のフィッティング
url:https://qiita.com/y-yamamoto-snt/items/bd12300e156dddcd0912



