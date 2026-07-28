import numpy as np

#logistic regression

class logisticRegression():
  def __init__(self, learningRate, iterations):
    self.learningRate = learningRate
    self.iterations = iterations


  def fit(self, X, Y):
    self.m, self.n = X.shape
    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    #implementing gradient descent
    for i in range(self.iterations):
      self.update_weights()

  def update_weights(self):

    #calculate gradients
    y_pred = self.predict(self.X)
    dw = (1/self.m)*np.dot(self.X.T, (y_pred - self.Y))
    db = (1/self.m)*np.sum(y_pred - self.Y)

    #updating weights

    self.w = self.w - self.learningRate*dw
    self.b = self.b - self.learningRate*db


  def predict(self, X):

    z = (X.dot(self.w) + self.b)
    y_pred = 1 / (1 + np.exp(-z))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred
