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
data = yf.download(ticker, start='2020-01-01', end='2024-12-30')

# Step 2: Feature Engineering
# Calculate Daily Returns
data['Returns'] = data['Adj Close'].pct_change()

# Calculate Volatility (rolling standard deviation of returns)
data['Volatility'] = data['Returns'].rolling(window=20).std()

# Calculate RSI (Relative Strength Index)
data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close'], window=14).rsi()

# Calculate Moving Averages
data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
data['EMA_20'] = data['Adj Close'].ewm(span=20, adjust=False).mean()

# Drop rows with NaN values resulting from calculations
data.dropna(inplace=True)

# Step 3: Prepare Data for HMM
features = data[['Returns', 'Volatility', 'Volume', 'RSI', 'SMA_20', 'EMA_20']]  # Feature selection

# Normalize Features
scaler = StandardScaler()
X = scaler.fit_transform(features)

# Step 4: Implement Gaussian HMM
# Initialize HMM with 4 Hidden States
model = GaussianHMM(n_components=4, covariance_type='full', n_iter=1000, random_state=42)

# Fit the Model
model.fit(X)

# Predict Hidden States
hidden_states = model.predict(X)

# Step 5: Add Hidden States to Data
data['Hidden_State'] = hidden_states

# Analyze Feature Means for Each State
state_means = data.groupby('Hidden_State').mean()

# Label Hidden States
state_labels = {
    0: 'Bullish',
    1: 'Bearish',
    2: 'Sideways Market',
    3: 'Volatility Change'
}

# Map Hidden States to Labels
data['Market_State'] = data['Hidden_State'].map(state_labels)

# Step 6: Evaluate Model Performance
log_likelihood = model.score(X)
print(f'Log Likelihood: {log_likelihood}')

# Step 7: Visualize Results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Adj Close'], label='Adjusted Close Price')

# Highlight Different Market States
for state, label in state_labels.items():
    plt.scatter(
        data.index[data['Hidden_State'] == state],
        data['Adj Close'][data['Hidden_State'] == state],
        label=label,
        marker='o',
        alpha=0.6
    )

plt.title(f'{ticker} Stock Price with Hidden Market States')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True)
plt.show()
