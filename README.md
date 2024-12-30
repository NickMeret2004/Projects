Project Overview

Welcome to the Hidden Markov Model (HMM) project for stock market prediction.

This project aims to predict hidden states of the stock market—such as Bullish and Bearish conditions—based on observed 
stock movements represented as discrete categories (e.g., Up, Down, or Flat).

The code utilizes the hmmlearn library, which implements machine learning algorithms to train and optimize the HMM.
It predicts the most likely sequence of hidden states based on the provided observations and evaluates the model’s performance.

This approach enables me to analyze and interpret underlying market trends that are not directly observable, offering insights into market behavior.


Code:

  Libraries:

    import numpy as np
    from hmmlearn.hmm import CategoricalHMM
    
      NumPy: Procides numerical and array manipiulation tools
      hmmlearn.hmm: implements machine learning algorithms for Hidden Markov Models, including CategoricalHMM, which handles discrete observations.


  observations: Encodes stock movements as categorical values

    observations = np.array([[0], [1], [2], [2], [0], [1], [2]])  
    
      0: Down (price decrease)
      1: Flat (no significant change)
      2: Up (price increase)


  Model: 

    model = CategoricalHMM(n_components=2, n_iter=100)
    
      CategoricalHMM: Suitable for modeling categorical (discrete) data.
      n_components = 2: Specifies two hidden states, representing Bullish and Bearish
      n_iter=100: Limits training to 100 iterations to optimize the model parameters.


  Training:

    model.fit(observations)

      Uses machine learning algos to estimate transition probabilities between hidden states.
      Leanrs emission probabilities, which link observations to hidden states.


  Predicting:
  
    hidden_states = model.predict(observations)
    print("Predicted Hidden States:", hidden_states)

      predict(): Predicts the most likely sequence of hidden states corresponding to the observations.
      Output: Outputs hidden state labels (ex: 0 for Bearish and 1 for Bullish) for each observation in the sequence.


  Model Performace Evaluation:

    log_prob = model.score(observations)
    print("Log Probability:", log_prob)

      score(): Computes the log-likelihood of the observations given the model.
      Log Probability: Provides a measure of how well the model fits the observed data. Higher values indicate better performance.
      
    
    
