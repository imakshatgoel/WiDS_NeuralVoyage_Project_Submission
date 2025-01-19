from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
#splitting data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
def BCELoss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    bce = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return bce
def sigmoid(x):
  return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.01, iters=1000): #lr (learning rate) & iters (iterations) could be anything of your choice
      self.weights = None
      self.bias = None
      self.lr = lr
      self.iters = iters

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.iters):
            y_pred = np.dot(X, self.weights) + self.bias
            y_pred_true = sigmoid(y_pred)
            dw = (-2 / n_samples) * np.dot(X.T, (y-y_pred))
            db = (-2 / n_samples) * np.sum(y-y_pred)
            self.weights-=(self.lr)*dw
            self.bias-=(self.lr)*db
      

    def predict(self, X):
      y_pred_proba = sigmoid(np.dot(X, self.weights) + self.bias)
      return [1 if prob > 0.5 else 0 for prob in y_pred_proba]
    
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
bce = BCELoss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Binary Cross Entropy Loss:{bce:.2f}")
print(f"Accuracy: {accuracy:.2f}")




