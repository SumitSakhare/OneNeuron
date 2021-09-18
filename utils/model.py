


class Perceptron:
  def __init__ (self, eta, epoch):
    self.weights= np.random.randn(3)*1e-4
    print(f"Initial weignts before training:\n{self.weights}")
    self.eta= eta
    self.epoch=epoch


  def Activationfunction(self, inputs, weights):
    z= np.dot(inputs, weights)
    return np.where(z>0,1,0)

  def fit(self, X, y):
    self.X= X
    self.y= y

    X_with_bias=np.c_[self.X, -np.ones((len(self.X),1))]
    print(f"X with bias: \n{X_with_bias}")

    for epoch in range(self.epoch):
      print("--"*10)
      print(f"For epoch: {epoch}")
      print("--"*10)

      y_hat= self.Activationfunction(X_with_bias, self.weights)
      print(f"Predicted value after forward pass:\n{y_hat}")
      self.error=self.y-y_hat
      print(f"error:\n{self.error}")
      self.weights=self.weights + self.eta * np.dot(X_with_bias.T, self.error)
      print(f"Updated wieghts after epoch:\n{epoch}:\n{self.weights}")

    

  def pridict(self, X):
    X_with_bias= np.c_[X, -np.ones((len(X), 1))]
    return self.Activationfunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss= np.sum(self.error)
    print(f"total loss: \n{total_loss}")
    return total_loss

