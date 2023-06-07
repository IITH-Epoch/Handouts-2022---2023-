import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost(object):
    def __init__(self) -> None:
        self.hypothesis = None
        self.hypothesis_weights = None

    def train(self,X, y, num_iteration):
        """
        Input
        --------------------------------
        X: ndarray of shape (Num,features)
        y: ndarray of shape (num,)
        num_itteration: int
        --------------------------------

        Output:
        --------------------------------
        hist: list
        --------------------------------
        """
        n = X.shape[0]
        w = np.full(n, 1/n)
        self.hypothesis = []
        self.hypothesis_weights = []
        hist = []
        for t in range(num_iteration):
            # training a model
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X,y,w)
            self.hypothesis.append(stump)
            # finding predicted value
            y_pred = stump.predict(X)
            # finding wieghted error
            err = np.sum(w*(y_pred != y))/np.sum(w)
            # finding alpha
            alpha = (1/2) * np.log((1-err)/err)
            self.hypothesis_weights.append(alpha)
            #updating value of w
            w = w*np.exp(-alpha*y*y_pred)
            # conputing the loss
            loss = np.mean(w)
            hist.append(loss)
        return hist

    def predict(self,X):
        """
        Input
        --------------------------------
        X: ndarray of shape (Num,features)
        --------------------------------

        Output:
        --------------------------------
        y: ndarray of shape (Num,)
        --------------------------------
        """
        num = X.shape[0]
        y = np.zeros(num)
        for alpha,h in zip(self.hypothesis_weights,self.hypothesis):
            y += alpha*(h.predict(X))
        pos = (y >= 0).astype("int") 
        neg = (y < 0).astype("int")
        y = pos - neg
        return y
