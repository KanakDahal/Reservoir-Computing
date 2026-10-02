import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1. Download Nifty 50 Data & Compute Realized Volatility
df = yf.download("^NSEI", start="2010-01-01", auto_adjust=True, progress=False)
close = df["Close"].squeeze().dropna()
r = 100 * np.diff(np.log(close.values))

# 5-day rolling standard deviation of returns (Realized Volatility in %)
vol_series = pd.Series(r, index=close.index[1:]).rolling(5).std().dropna()
vol = vol_series.values
dates = vol_series.index

u = vol[:-1]  # Input: Today's volatility (vol_t)
y = vol[1:]   # Target: Tomorrow's volatility (vol_{t+1})

# Train/Test Split (discard first 50 days for reservoir warmup, split at 3500)
warmup, split = 50, 3500
y_train, y_test = y[warmup:split], y[split:]
test_dates = dates[split + 1:]

# --- STEP 1: ERC Hyperparameters ---
M = 10               # Number of sub-reservoirs in the ensemble
N = 30               # Neurons per sub-reservoir
leak = 0.3           # Leaking rate (memory decay)
spectral_radius = 0.90
alpha = 10.0         # Ridge regularization (higher for noisy market data)

np.random.seed(42)
ensemble_preds = np.zeros((M, len(y_test)))

# Standardize input u using training statistics so tanh() doesn't saturate
u_mean, u_std = u[warmup:split].mean(), u[warmup:split].std()
u_norm = (u - u_mean) / u_std

# --- STEP 2: Loop Over M Independent Sub-Reservoirs ---
for m in range(M):
    # A. Random input and recurrent reservoir weights
    W_in = np.random.uniform(-0.5, 0.5, (N, 1))
    W_res = np.random.uniform(-0.5, 0.5, (N, N))
    
    # Scale W_res to desired spectral radius
    eig_max = np.max(np.abs(np.linalg.eigvals(W_res)))
    W_res *= (spectral_radius / eig_max)
    
    # B. Drive reservoir m with normalized volatility
    R = np.zeros((len(u_norm), N))
    r_state = np.zeros(N)
    for t in range(len(u_norm)):
        r_state = (1 - leak) * r_state + leak * np.tanh(W_res @ r_state + W_in[:, 0] * u_norm[t])
        R[t] = r_state
        
    # Feature matrix: [1 (bias), u_norm, reservoir states R]
    F = np.column_stack([np.ones(len(u_norm)), u_norm, R])
    F_train, F_test = F[warmup:split], F[split:]
    
    # C. Fit Ridge Regression readout for sub-reservoir m
    I = np.eye(F_train.shape[1])
    I[0, 0] = 0.0  # Do not penalize the intercept/bias term
    W_out = np.linalg.solve(F_train.T @ F_train + alpha * I, F_train.T @ y_train)
    
    # Store predictions for sub-reservoir m
    ensemble_preds[m] = F_test @ W_out

# --- STEP 3: Ensemble Aggregation ---
erc_predictions = ensemble_preds.mean(axis=0)

# Evaluation Metrics (RMSE & Correlation)
single_rmse = np.sqrt(np.mean((y_test - ensemble_preds[0]) ** 2))
erc_rmse = np.sqrt(np.mean((y_test - erc_predictions) ** 2))
corr = np.corrcoef(y_test, erc_predictions)[0, 1]

print(f"Single Reservoir RMSE : {single_rmse:.4f}%")
print(f"Ensemble (ERC) RMSE   : {erc_rmse:.4f}%")
print(f"Forecast Correlation  : {corr:.4f}")

# --- STEP 4: Plot Actual vs. ERC Predicted Volatility ---
plt.figure(figsize=(12, 5))
# Plot the first 150 test days for clear visual comparison
n_plot = 150
for m in range(M):
    plt.plot(test_dates[:n_plot], ensemble_preds[m, :n_plot], color="gray", alpha=0.25, lw=1, label="Sub-Reservoirs" if m == 0 else "")

plt.plot(test_dates[:n_plot], y_test[:n_plot], label="Actual 5-Day Volatility (%)", color="steelblue", lw=2)
plt.plot(test_dates[:n_plot], erc_predictions[:n_plot], label=f"ERC Mean Prediction (M={M})", color="crimson", linestyle="--", lw=2)
plt.title(f"Nifty 50 Volatility Forecasting with Ensemble Reservoir Computing | Test Corr: {corr:.3f}")
plt.xlabel("Date")
plt.ylabel("Realized Volatility (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()