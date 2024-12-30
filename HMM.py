#Hidden Markov Model
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import ta
import matplotlib.pyplot as plt

# Step 1: Download NVDA Stock Data (Past 4 Years)
ticker = 'NVDA'  # NVIDIA's stock ticker symbol
data = yf.download(ticker, start='2024-01-01', end='2024-12-30')

# Step 2: Feature Engineering
# Calculate Daily Returns
data['Returns'] = data['Close'].pct_change()

# Calculate Volatility (rolling standard deviation of returns)
data['Volatility'] = data['Returns'].rolling(window=20).std()

# Calculate RSI (Relative Strength Index)
data['RSI'] = ta.momentum.RSIIndicator(data['Close'].squeeze(), window=14).rsi()

# Calculate Moving Averages
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

# *** NEW: Momentum Indicator ***
data['Momentum'] = ta.momentum.ROCIndicator(data['Close'].squeeze(), window=10).roc()

# *** NEW: Add MACD and Signal Line ***
macd = ta.trend.MACD(data['Close'].squeeze())  # Ensure Close is 1D
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

# Drop rows with NaN values resulting from calculations
data.dropna(inplace=True)

# Step 3: Prepare Data for HMM
features = data[['Returns', 'Volatility', 'Volume', 'RSI', 'SMA_20', 'EMA_20']]  # Feature selection

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Step 4: Implement Gaussian HMM
# *** NEW: Increased Hidden States for More Granularity ***
model = GaussianHMM(n_components=6, covariance_type='full', n_iter=1000, random_state=42)

# *** NEW: Force Better Initialization for State Transitions ***
model.transmat_ = np.array([
    [0.85, 0.10, 0.03, 0.02],  # Bullish
    [0.10, 0.85, 0.03, 0.02],  # Bearish
    [0.05, 0.05, 0.85, 0.05],  # Sideways
    [0.02, 0.02, 0.05, 0.91]   # Volatility Change
])

# Fit the Model
model.fit(X)

# Predict Hidden States
hidden_states = model.predict(X)

# Step 5: Add Hidden States to Data
data['Hidden_State'] = hidden_states

# Analyze Feature Means for Each State
state_means = data.groupby('Hidden_State').mean()

# *** NEW: Label Hidden States ***
state_labels = {
    0: 'Bullish',
    1: 'Bearish',
    2: 'Sideways Market',
    3: 'Volatility Change',
    4: 'Consolidation',  # *** NEW State Label ***
    5: 'Breakout'        # *** NEW State Label ***
}

# Map Hidden States to Labels
data['Market_State'] = data['Hidden_State'].map(state_labels)

# Step 6: Evaluate Model Performance
log_likelihood = model.score(X)
print(f'Log Likelihood: {log_likelihood}')

# Step 7: Visualize Results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price')

# *** NEW: Step 8 - Plot Hidden State Probabilities ***
posterior_probs = model.predict_proba(X)

plt.figure(figsize=(14, 7))
for i, label in state_labels.items():
    smoothed_probs = pd.DataFrame(posterior_probs).rolling(window=7).mean()  # Smooth probabilities
for i, label in state_labels.items():
    plt.plot(data.index, smoothed_probs[i], label=label)

plt.title('Hidden State Probabilities Over Time')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
