# Commute-Time-Estimator
A single-layer neural network using PyTorch to accept a distance-input and output a predicted commute time. Program reads prior distance–time data and trains a linear regression model to approximate the relationship between the distance and travel duration to generate a prediction.


- Model can now predict delivery times for any distance using given data (must track and use your personal data values for time and distance)
- Equation: Time = Weight * Distance + Bias
- Weight = meaning each additional mile will increase delivery by [Weight] mins
- Bias = representing base time for each delivery
