import numpy as np

class LogisticRegression(object):
  """
  Simple implementation of logistic regression classifier

  Parameter(s):

    penalty : string(default 'l2'): regularization penalty

    lmbda : float(default 1): regularization value

    lrate : float(default 0.1): learning rate
    
    epochs : int(default = 100): number of times the gradient descent should run

    normalize : boolean(default True): if the data has to be normalized first, (min-max normalization)
  """

  def __init__(self, penalty='l2', lmbda=1, lrate=0.1, epochs=100, normalize=True):
    assert penalty in ['l1', 'l2', None], "Penalty can be 'l1','l2' or None"

    self.penalty = penalty
    self.lmbda = lmbda
    self.lrate = lrate
    self.epochs = epochs
    self.normalize = normalize
    self.weights = None

  def __sigmoid(self, z):
    """
    Returns the sigmoid value of the given matrix

      Parameter(s):
        z : given matrix
    """
    
    return 1 / (1 + np.exp(-z, dtype=np.float128))

  def __normalization(self, X):
    """
    Normalizes the given data using min-max scaling
    
      Parameter(s):
        X : given matrix
    """
    return (X-X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

  def __costGradFunction(self, X, y, weights):
    """
    Returns cost value as per logistic loss function and calculated gradient of weight
    as per the set penalty
    
      Parameter(s):
        X : data set (data samples, num of features)
        y : true labels (data samples, 1)
        weights : feature-wise weights
    """
    # calculating costs
    m = len(y)
    pred = self.__sigmoid(X @ weights)
    cost = (1/m)*np.sum((-y * np.log(pred)) - ((1-y)*np.log(1-pred)))

    if self.penalty == 'l1':
      reg_cost = self.lmbda/(2*m) * np.sum(np.abs(weights))
    elif self.penalty == 'l2':
      reg_cost = self.lmbda/(2*m) * np.sum(weights**2)
    else:
      reg_cost = 0
    
    # calculating gradients
    if self.penalty == 'l1':
      grad = 1/m * (X.T @ (pred - y)) + (self.lmbda/m)*((weights+1e-5)/np.abs(weights+1e-5))
    elif self.penalty == 'l2':
      grad = 1/m * (X.T @ (pred - y)) + (self.lmbda/m)*weights
    else:
      grad = 1/m * (X.T @ (pred - y))
    
    return cost+reg_cost, grad

  def predict(self, X_test, pi = 0.5):
    """
    Returns the predicted labels for test data based on some probability threshold
    
      Parameter(s):
        X_test : test data set (test data samples, num of features)
    """
    assert self.weights is not None, "Classifier has to be trained before prediction"
    if self.normalize:
      X_test = self.__normalization(X_test)

    X_test = np.c_[np.ones([np.shape(X_test)[0], 1]), X_test]
    probs = self.__sigmoid(X_test @ self.weights)
    y_pred = (probs>=pi).astype(int)
    return y_pred

  def fit(self, X_train, y_train):
    """
    Finds the best fitting hyperplane based on the train set
    
      Parameter(s):
        X_train : train dataset (train data samples, num of features)
        y_train : (n_samples)
    """
    assert len(np.unique(y_train))==2, "This is just a binary implementation. Multi-class not allowed."
    if self.normalize:
      X_train = self.__normalization(X_train)

    X_train = np.c_[np.ones([np.shape(X_train)[0], 1]), X_train]
    weights = np.zeros((np.shape(X_train)[1], 1))

    for i in range(self.epochs):
      cost, grad = self.__costGradFunction(X_train,y_train,weights)
      weights = weights - (self.lmbda * grad)

    self.weights = weights

  def evaluate(self, X_test, y_test):
    """
    Evaluates the classifier for each test instances based on the true and predicted classes
    
    Parameter(s):
      X_test : test dataset (test data samples, num of features)
      y_test : true labels
    """
    y_pred = self.predict(X_test)
    return round(np.sum(y_test == y_pred)/len(y_pred), 3)

if __name__ == "__main__":
  train_data = np.genfromtxt('./data/Pima_Indian_Patents_train.csv',delimiter=',')
  test_data = np.genfromtxt('./data/Pima_Indian_Patents_test.csv',delimiter=',')

  X_train, y_train = train_data[:,:-1], train_data[:,-1].reshape(-1,1)
  X_test, y_test = test_data[:,:-1], test_data[:,-1].reshape(-1,1) 

  lr = LogisticRegression(lmbda=10)
  lr.fit(X_train, y_train)
  train_accuracy = lr.evaluate(X_train, y_train)
  print("Train Accuracy : %f" %train_accuracy)
  test_accuracy = lr.evaluate(X_test, y_test)
  print("Test Accuracy : %f" %test_accuracy)
