Project Overview

This project uses a Gaussian Hidden Markov Model (HMM) to predict hidden states in NVIDIA’s (NVDA) stock market trends over the past 4 years.

The model incorporates real stock data and applies feature engineering to extract meaningful patterns, such as returns, volatility, volume, and technical indicators like RSI and moving averages.

It predicts hidden market states, including:
	•	Bullish (positive trends)
	•	Bearish (negative trends)
	•	Sideways Market (neutral or consolidating trends)
	•	Volatility Change (sudden fluctuations)

The code leverages machine learning algorithms from the hmmlearn library to uncover underlying market regimes and visualize trends over time.


Code:

  Libraries:

    import yfinance as yf
    import numpy as np
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    import ta
    import matplotlib.pyplot as plt
    
      •	yfinance: Fetches historical stock data from Yahoo Finance.
	    •	NumPy: Provides numerical and array manipulation tools.
	    •	Pandas: Handles dataframes and simplifies data manipulation.
	    •	GaussianHMM: Models continuous data with Hidden Markov Models.
	    •	StandardScaler: Standardizes data to ensure equal feature contribution.
	    •	ta: Computes technical indicators like RSI and moving averages.
	    •	Matplotlib: Visualizes stock prices and predicted hidden states.


  Fetching Historical Data

    data = yf.download('NVDA', start='2020-01-01', end='2024-12-30')

    	•	Downloads 4 years of historical data for NVDA stock.
	    •	Includes fields like Open, Close, Volume, and Adjusted Close prices.


  Feature Engineering 

    data['Returns'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close'], window=14).rsi()
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Adj Close'].ewm(span=20, adjust=False).mean()
    data.dropna(inplace=True)
    
      •	Returns: Percentage change in closing prices.
	    •	Volatility: Rolling standard deviation of returns (measures risk).
	    •	RSI (Relative Strength Index): Identifies overbought/oversold conditions.
	    •	SMA/EMA: Moving averages to capture trends over 20 days.
	    •	Drop NaN: Removes rows with missing values caused by rolling calculations.


  Prepare Data for HMM

    features = data[['Returns', 'Volatility', 'Volume', 'RSI', 'SMA_20', 'EMA_20']]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

      •	Feature Selection: Uses engineered features to predict market states.
	    •	Standardization: Normalizes data for equal scaling of features.


  Gaussian Hidden Markov Model (HMM)

    model = GaussianHMM(n_components=4, covariance_type='full', n_iter=1000, random_state=42)
    model.fit(X)

      •	GaussianHMM: Models continuous data distributions.
	    •	n_components = 4: Specifies 4 hidden states:
	      •	Bullish
	      •	Bearish
	      •	Sideways Market
	      •	Volatility Change
	    •	covariance_type=‘full’: Allows flexible covariance structures for better fitting.
	    •	n_iter = 1000: Runs up to 1000 iterations to optimize model parameters.
	    •	fit(): Trains the model to estimate transition probabilities and emission probabilities.


  Predict Hidden States

    hidden_states = model.predict(X)
    data['Hidden_State'] = hidden_states

      •	predict(): Predicts the most likely hidden states for each observation.
	    •	Hidden States: Assigns labels to indicate market conditions.


  Label Hidden States

    state_labels = {
    0: 'Bullish',
    1: 'Bearish',
    2: 'Sideways Market',
    3: 'Volatility Change'
    }
    data['Market_State'] = data['Hidden_State'].map(state_labels)

      •	Maps numerical states to descriptive labels for easier interpretation.
	    •	Market States: Categorizes periods as Bullish, Bearish, Sideways, or Volatile.


  Model Perfomance Evaluation

    log_likelihood = model.score(X)
    print(f'Log Likelihood: {log_likelihood}')

      •	score(): Calculates the log-likelihood of the data under the model.
	    •	Log Likelihood: Evaluates model fit—higher values indicate better performance.


  Visualize Results

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Adj Close'], label='Adjusted Close Price')

    for state, label in state_labels.items():
      plt.scatter(data.index[data['Hidden_State'] == state],
                data['Adj Close'][data['Hidden_State'] == state],
                label=label, marker='o', alpha=0.6)

    plt.title('NVDA Stock Price with Hidden Market States')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

      •	Plots stock prices with colored markers indicating market states.
	    •	Allows visual inspection of how the model identifies trends and volatility.


  Output Ex:

    Log Likelihood: -524.29

      •	Graph: Displays Bullish, Bearish, Sideways, and Volatility Change states over time,       plotted against the stock price.
      
    
    
