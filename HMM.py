#Hidden Markov Model
import numpy as np
from hmmlearn.hmm import CategoricalHMM

# Define Observations (Discrete Stock Movements)
observations = np.array([[0], [1], [2], [2], [0], [1], [2]])  # 0=Down, 1=Flat, 2=Up

# Initialize Categorical HMM
model = CategoricalHMM(n_components=2, n_iter=100)  # 2 Hidden States (Bullish, Bearish)

# Train the Model
model.fit(observations)

# Predict Hidden States
hidden_states = model.predict(observations)
print("Predicted Hidden States:", hidden_states)

# Evaluate Model
log_prob = model.score(observations)
print("Log Probability:", log_prob)
