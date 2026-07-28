import numpy as np

#linear regression model

class linearRegression():
  #initiating the parameters (learning rate and no. of iterations)
  def __init__(self, learning_rate, iterations):
    self.learning_rate = learning_rate
    self.iterations = iterations

  def fit(self,X, Y):
    #number of traing examples and number of features
    self.m, self.n = X.shape #number of rows and columns

    #initiating the weights and bias
    self.w= np.zeros(self.n)
    self.b= 0
    self.X=X
    self.Y = Y

    #implementing gradient descent
    for i in range(self.iterations):
      self.update_weights()

  def update_weights(self):

    Y_prediction = self.predict(self.X)

    #calculate gradients

    dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
    db = -(2 * np.sum(self.Y - Y_prediction)) / self.m

    #updating the weights

    self.w = self.w - self.learning_rate*dw
    self.b = self.b - self.learning_rate*db

  def predict(self, X):

    return X.dot(self.w) + self.b
