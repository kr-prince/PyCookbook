import numpy as np

class kNearestNeighbours(object):
  """
  Simple implementation of k nearest neighbour classifier

  Parameter(s):

    neighbours : int: how many neighbours to consider

    weighted : boolean(default False): if weighted the score is inverse of distance else score is 1 
    
    normalize : boolean(default True): if the data has to be normalized first, (min-max normalization)
    
    dist_func : string(default euclidean): "cosine" or "euclidean"
  """

  def __init__(self, kvalue, weighted=False, normalize=True, dist_metric="euclidean"):
    assert dist_metric in ['euclidean','cosine'],"Distance metric can be 'euclidean'/'cosine'"

    self.kvalue = kvalue 
    self.weighted = weighted
    self.normalize = normalize
    self.dist_metric = dist_metric

  def euclidean_dist(self, p1, p2):
    """
    Calculates euclidean distance between two given points
      
      Parameter(s):
        p1 : vector or matrix
        p2 : vector
      
      Returns single distance value in case of vector or list of distances for matrix
    """
    
    p1 = np.expand_dims(p1, axis=0) if p1.ndim <= 1 else p1
    p2 = np.expand_dims(p2, axis=0) if p2.ndim <= 1 else p2
    assert p1.shape[1]==p2.shape[1], "Dimension mismatch (%d != %d)" % (p1.shape[1], p2.shape[1])
    
    dist = np.sqrt(np.sum(np.square(p1 - p2), axis=1))
    return dist

  def cosine_dist(self, p1, p2):
    """
    Calculates cosine distance between two given points
    
      Parameter(s):
        p1 : vector or matrix
        p2 : vector
      
      Returns single distance value in case of vector or list of distances for matrix
    """
    
    p1 = np.expand_dims(p1, axis=0) if p1.ndim <= 1 else p1
    
    cosine_sim = np.dot(p1, p2.T)/np.multiply(np.linalg.norm(p1, axis=1), np.linalg.norm(p2))
    return 1.0 - cosine_sim

  def normalization(self, X):
    """
    Normalizes the given data using min-max scaling
    
      Parameter(s):
        X : given matrix
    """
    
    return (X-X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

  def train(self, X_train, y_train):
    """
    Takes training data and saves for prediction
    
      Parameter(s):
        X_train : training data features matrix
        y_train : training data classes
    """
    
    self.X_train = self.normalization(X_train) if self.normalize else X_train
    self.y_train = y_train

  def predict(self, X_test):
    """
    Predicts the classes for each test instances based on the train set
    
     Parameter(s):
        X_test : given matrix of features to predict
    """
    
    y_pred = []
    dist_func = self.euclidean_dist if self.dist_metric=='euclidean' else self.cosine_dist
    X_test = self.normalization(X_test) if self.normalize else X_test
    for i,vector in enumerate(X_test):
      distances = dist_func(self.X_train, vector)
      neighbours = np.argsort(distances)[:self.kvalue]
      
      classes = {}
      for ni in neighbours:
        classes[self.y_train[ni]] = classes.get(self.y_train[ni], 0.0) + (1 if not self.weighted else 1/distances[ni])
      votes = sorted(tuple(classes.items()), key=lambda x:x[1], reverse=True)
      y_pred.append(votes[0][0])
    return y_pred

  def evaluate(self, X_test, y_test):
    """
    Evaluates the classifier for each test instances based on the true and predicted classes
    
    Parameter(s):
      X_test : given matrix of features
      y_test : given vector of class labels
    """

    y_pred, total = self.predict(X_test), y_test.shape[0]
    hit = sum([1 if y_test[i]==y_pred[i] else 0 for i in range(total)])
    return round((hit * 100)/total, 2)

if __name__ == "__main__":
  train_data = np.genfromtxt('./data/Pima_Indian_Patents_train.csv',delimiter=',')
  test_data = np.genfromtxt('./data/Pima_Indian_Patents_test.csv',delimiter=',')

  X_train, y_train = train_data[:,:-1], train_data[:,-1] 
  X_test, y_test = test_data[:,:-1], test_data[:,-1] 

  knn = kNearestNeighbours(5)
  knn.train(X_train, y_train)
  accuracy = knn.evaluate(X_test, y_test)
  print("Accuracy : %f" %accuracy)
