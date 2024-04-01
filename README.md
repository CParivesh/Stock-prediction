
# Overview

This repository contains implementations of two stock prediction algorithms: Stacked Long Short-Term Memory (LSTM) and Support Vector Regression (SVR) vs Linear Regression. These algorithms aim to forecast stock prices based on historical data, enabling investors and traders to make informed decisions.

## Algorithms

### Stacked LSTM
The Stacked LSTM algorithm is a type of recurrent neural network (RNN) architecture designed for sequence prediction tasks. It utilizes multiple LSTM layers stacked on top of each other to capture complex temporal dependencies in the input data. The model is trained on historical stock price data and learns to predict future price movements.

SVR vs Linear Regression
This algorithm compares the performance of Support Vector Regression (SVR) and Linear Regression models for stock price prediction. SVR is a machine learning algorithm that finds the optimal hyperplane to minimize prediction errors, while Linear Regression fits a linear model to the data. By evaluating both models, we can determine which approach yields better results for forecasting stock prices.

### Usage

Data Preparation: Ensure that you have historical stock price data in a suitable format for training the algorithms. This data should include features such as opening price, closing price, volume, etc.
Model Training: Choose the algorithm you want to use (Stacked LSTM or SVR vs Linear Regression) and train the model using the provided training data.
Evaluation: Evaluate the performance of the trained model using testing data or cross-validation techniques. Compare the predicted stock prices with the actual prices to assess the accuracy of the predictions.
Deployment: Once satisfied with the model's performance, deploy it for real-world stock price prediction tasks. Monitor its performance regularly and make adjustments as necessary.

### Requirements

Python 3.x
Libraries: TensorFlow (for Stacked LSTM), Scikit-learn (for SVR vs Linear Regression), Pandas, NumPy, Matplotlib

### License

This project is licensed under the General Public.

### Acknowledgements

TensorFlow
Scikit-learn
Pandas
NumPy
Matplotlib
Author

Parivesh
