Hidden Markov Model 

Hello and welcome to my latest project! I started this project with the goal of deepening my understanding of Python and its libraries,
while also exploring stock market movements, predicting hidden states in stock prices, downloading market data, and applying feature 
engineering techniques.
	
I would like to give credit to where it is due, which is to a youtube video I stumbled upon by MoonDev 
https://www.youtube.com/watch?v=JwGfdnnyUAs . This is just a fantastic video showing uncut footage of someone just like me slowly 
learning how to use new AI tools to predict as much as possible about stock market movements. I would also like to share a video that 
originally helped me understand the idea of Hidden Markov Models and how they work “under the hood” https://www.youtube.com/watch?v=RWkHJnFj5rY . This video, by the channel Normalized Nerd, was crucial in my understanding of the Hidden Markov Model and its process. 
Without these people and their channels, I'm not sure if I ever would have stumbled upon this great topic, and for them I am truly 
grateful.

Main Goal: the goal of this project wasn’t to create a perfect model, but to create a framework for which I 
can constantly improve as my coding knowledge and experience grows with time. Now let's get into the details of my project:



This project implements a Gaussian Hidden Markov Model (HMM) to analyze and predict hidden states in NVIDIA’s (NVDA) stock price movements over the past year.

Project Goals
	1.	Identify hidden states based on historical stock data to better understand market behavior.
	2.	Classify market conditions into states like: Bullish (Uptrend), Bearish (Downtrend), Sideways Market (Neutral/Consolidation), Volatility Change (Sharp moves in price), Consolidation (Stable patterns before breakout), and Breakout (Sudden directional moves).
	3.	Enhance feature engineering to capture signals such as momentum, volatility, and moving averages.
	4.	Improve state predictions by adding smoothed probabilities for clearer trend analysis.


Libraries

import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import ta
import matplotlib.pyplot as plt

Library Breakdown:
yfinance fetches stock price data from Yahoo Finance. NumPy provides tools for numerical computation and matrix operations, which are essential for HMMs. Pandas allows easy handling and manipulation of stock data in a tabular format. GaussianHMM implements Hidden Markov Models with Gaussian (continuous) emissions. Suitable for analyzing numerical features like returns and volatility. StandardScaler standardizes input features to have zero mean and unit variance, preventing larger values (e.g., volume) from dominating smaller ones (e.g., RSI). ta (Technical Analysis) computes financial indicators like RSI, MACD, and moving averages to identify trends. Matplotlib visualizes stock price movements and hidden state probabilities.


Data Download

data = yf.download(ticker, start=‘2024-01-01’, end=‘2024-12-30’)

This downloads daily stock price data for NVIDIA (NVDA) between January 1, 2024, and December 30, 2024. It includes columns like Open, High, Low, Close, Adj Close, and Volume. The Close price is primarily used for feature engineering because it reflects the final price of a trading day.


Feature Engineering

Daily Returns and Volatility:

data[‘Returns’] = data[‘Close’].pct_change()
data[‘Volatility’] = data[‘Returns’].rolling(window=20).std()

Returns measure the percentage change in price from one day to the next. Returns track daily performance to identify trends and momentum. Volatility calculates the rolling standard deviation of returns over a 20-day window. It measures the market’s risk or variability—higher volatility often precedes trend reversals or breakouts.


Momentum Indicators:

data[‘Momentum’] = ta.momentum.ROCIndicator(data[‘Close’].squeeze(), window=10).roc()

Momentum (ROC) calculates the Rate of Change over the last 10 days. It detects whether the stock is accelerating or decelerating, helping identify shifts in momentum that might indicate trend changes.


MACD and Signal Line:

macd = ta.trend.MACD(data[‘Close’].squeeze())
data[‘MACD’] = macd.macd()
data[‘MACD_Signal’] = macd.macd_signal()

MACD (Moving Average Convergence Divergence) detects trend reversals by comparing two moving averages: Fast Line (12-day EMA) and Slow Line (26-day EMA). The Signal Line (9-day EMA) smooths out the MACD. Crossovers between the MACD and Signal Line indicate buy or sell signals.


RSI (Relative Strength Index):

data[‘RSI’] = ta.momentum.RSIIndicator(data[‘Close’].squeeze(), window=14).rsi()

RSI measures momentum by comparing the magnitude of recent gains to losses over 14 days. RSI > 70 indicates Overbought conditions (may lead to a pullback), while RSI < 30 indicates Oversold conditions (may lead to a rally).


Moving Averages:

data[‘SMA_20’] = data[‘Close’].rolling(window=20).mean()
data[‘EMA_20’] = data[‘Close’].ewm(span=20, adjust=False).mean()

SMA (Simple Moving Average) smooths price data by averaging over 20 days. EMA (Exponential Moving Average) gives more weight to recent prices, making it more responsive. Moving averages highlight long-term trends and support/resistance levels.


Hidden Markov Model (HMM)

Model Setup:

model = GaussianHMM(n_components=6, covariance_type=‘full’, n_iter=1000, random_state=42)

This model uses 6 hidden states to capture trends such as Bullish, Bearish, Sideways, Volatility Change, Consolidation, and Breakout. Gaussian HMM handles continuous data by modeling each feature with a Gaussian distribution. n_iter=1000 trains the model for up to 1000 iterations for convergence.

Transition Probabilities:

model.transmat_ = np.array([
[0.85, 0.10, 0.03, 0.02],
[0.10, 0.85, 0.03, 0.02],
[0.05, 0.05, 0.85, 0.05],
[0.02, 0.02, 0.05, 0.91]
])

These transition probabilities control how likely the model is to stay in one state or switch to another. Higher values on the diagonal enforce stability, while lower values allow rare transitions.


Model Predictions

hidden_states = model.predict(X)

Predicts the most likely hidden states for each observation based on the model’s probabilities and learned parameters.


State Labels

state_labels = {
0: ‘Bullish’,
1: ‘Bearish’,
2: ‘Sideways Market’,
3: ‘Volatility Change’,
4: ‘Consolidation’,
5: ‘Breakout’
}

This dictionary maps numeric state outputs to descriptive labels for easy interpretation.


Model Evaluation

log_likelihood = model.score(X)
print(f’Log Likelihood: {log_likelihood}’)

Calculates the log-likelihood, which measures how well the model fits the data. Higher values indicate better performance.


Visualizations

Price Chart with Hidden States:

plt.figure(figsize=(14, 7))
plt.plot(data.index, data[‘Close’], label=‘Close Price’)

Visualizes stock price movements with hidden states overlaid.

State Probabilities (Smoothed):

posterior_probs = model.predict_proba(X)
smoothed_probs = pd.DataFrame(posterior_probs).rolling(window=7).mean()
for i, label in state_labels.items():
plt.plot(data.index, smoothed_probs[i], label=label)

Plots smoothed probabilities over time using a 7-day rolling average to highlight stable trends instead of noise.

plt.title(‘Hidden State Probabilities Over Time’)
plt.xlabel(‘Date’)
plt.ylabel(‘Probability’)
plt.legend()
plt.grid(True)
plt.show()


Output

The output consists of two graphs, one showing the time period and stock price over that time period, and the other displays the hidden state probabilities over time, based on the predictions of our Guassian Hidden Markov Model. With both of these we can see over time what our code believes to be hidden states of, in this case, the SP 500 (seen in the screenshots below). As we can see, over the past month the model has picked up a bearish setiment in the market as the year of 2024 ends and a new president elect coming into office. More recently, bearish setiment seems to be dropping and more of a consolidation phase seems to be setting in.
      
![Screenshot 2025-01-20 at 5 11 23 PM](https://github.com/user-attachments/assets/6c43d134-418f-4a6a-b8bd-f532bade09c9)
![Screenshot 2025-01-20 at 5 11 36 PM](https://github.com/user-attachments/assets/de55e4e5-c128-49f3-a386-c39c59b775a2)

    
