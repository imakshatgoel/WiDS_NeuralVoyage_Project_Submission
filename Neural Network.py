import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Numerical stability
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# Neural Network class
class NN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * 0.01
        self.B1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * 0.01
        self.B2 = np.zeros((1, self.output_size))

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1.T) + self.B1
        self.A1 = ReLU(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2.T) + self.B2
        self.A2 = softmax(self.Z2)
        return self.A2

    def one_hot(self, y):
        one_hot_matrix = np.zeros((y.size, self.output_size))
        one_hot_matrix[np.arange(y.size), y] = 1
        return one_hot_matrix

    def backward_propagation(self, X, y):
        m = X.shape[0]
        self.dZ2 = self.A2 - y
        self.dW2 = (1 / m) * np.dot(self.dZ2.T, self.A1)
        self.dB2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True)

        self.dZ1 = np.dot(self.dZ2, self.W2) * (self.Z1 > 0)
        self.dW1 = (1 / m) * np.dot(self.dZ1.T, X)
        self.dB1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True)

    def update_params(self):
        self.W1 -= self.learning_rate * self.dW1
        self.B1 -= self.learning_rate * self.dB1
        self.W2 -= self.learning_rate * self.dW2
        self.B2 -= self.learning_rate * self.dB2

    def get_predictions(self, X):
        A2 = self.forward_propagation(X)
        return np.argmax(A2, axis=1)

    def get_accuracy(self, X, y):
        predictions = self.get_predictions(X)
        return np.mean(predictions == y)

    def gradient_descent(self, X, y, iters=1000):
        y_one_hot = self.one_hot(y)
        for i in range(iters):
            self.forward_propagation(X)
            self.backward_propagation(X, y_one_hot)
            self.update_params()
            if i % 100 == 0:
                cost = self.cross_entropy_loss(y_one_hot)
                print(f"Iteration {i}, Cost: {cost}")

    def cross_entropy_loss(self, y_one_hot):
        m = y_one_hot.shape[0]
        log_probs = -np.log(self.A2[np.arange(m), np.argmax(y_one_hot, axis=1)])
        return np.sum(log_probs) / m

    def show_predictions(self, X, y, num_samples=10):
        random_indices = np.random.randint(0, X.shape[0], size=num_samples)
        for index in random_indices:
            sample_image = X[index, :].reshape((28, 28))
            plt.imshow(sample_image, cmap='gray')
            plt.title(f"Actual: {y[index]}, Predicted: {self.get_predictions(X[index:index+1])[0]}")
            plt.show()

# Load and prepare the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images
X_train = X_train.reshape((60000, -1))
X_test = X_test.reshape((10000, -1))

# Train the model
input_size = 784
hidden_size = 256
output_size = 10
learning_rate = 0.01

model = NN(input_size, hidden_size, output_size, learning_rate)
model.gradient_descent(X_train, Y_train, iters=1000)

# Test the model
accuracy = model.get_accuracy(X_test, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize predictions
model.show_predictions(X_test, Y_test, num_samples=5)
