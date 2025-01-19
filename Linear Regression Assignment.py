import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("C:\WiDS-Neural-Voyage\Week 2\HousingData.csv")

from sklearn.preprocessing import StandardScaler

X = data.drop(columns='MEDV').values    # All input features
y = data['MEDV'].values                 # Target variable

scaler = StandardScaler()
X = scaler.fit_transform(X)             # Normalizing the input data to avoid any overflows

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # specifying random state ensures that the random split is same everytime we run this command

# Drop rows with NaN or Inf
y_train = y_train[~np.isnan(X_train).any(axis=1)]
X_train = X_train[~np.isnan(X_train).any(axis=1)]

y_train = y_train[~np.isinf(X_train).any(axis=1)]
X_train = X_train[~np.isinf(X_train).any(axis=1)]

# Similarly for X_test
y_test = y_test[~np.isnan(X_test).any(axis=1)]
X_test = X_test[~np.isnan(X_test).any(axis=1)]

y_test = y_test[~np.isinf(X_test).any(axis=1)]
X_test = X_test[~np.isinf(X_test).any(axis=1)]

class LinearRegression:
    def __init__(self):
        # Initialize weights and bias
        self.weights = None
        self.bias = None

    def fit(self, X, y, lr=0.01, epochs=1000):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(epochs):
            y_pred = self.predict(X)
            dw = (-2 / n_samples) * np.dot(X.T, (y-y_pred))
            db = (-2 / n_samples) * np.sum(y-y_pred)
            self.weights-=lr*dw
            self.bias-=lr*db 
    

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Define the R2_score function
def R2_score(y_test, y_test_pred):
    y_mean = y_test.mean()
    return 1 - ((sum((y_test - y_test_pred) ** 2)) / (sum((y_test - y_mean) ** 2)))

# Predict on test data
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = model.mean_squared_error(y_test, y_test_pred)
r2 = R2_score(y_test, y_test_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and evaluate
y_test_pred_sklearn = lr.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_test_pred_sklearn)
r2_sklearn = r2_score(y_test, y_test_pred_sklearn)

print(f"Sklearn Model's Mean Squared Error: {mse_sklearn:.2f}")
print(f"Sklearn Model's R-squared Score: {r2_sklearn:.2f}")

# Compare weights and bias

print(f"Your Model's Weights: {model.weights}, Bias: {model.bias}")
print(f"Sklearn Model's Weights: {lr.coef_}, Bias: {lr.intercept_}")


