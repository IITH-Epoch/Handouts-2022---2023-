import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GBM(object):
    def __init__(self) -> None:
        self.alpha = None
        self.weak_learners = None

    def train(self,X,y,num_itr,alpha = 1):
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
        # Initialization
        self.weak_learners = []
        self.alpha = alpha
        hist = []
        # fitting data 
        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(X,y)
        self.weak_learners.append(stump)
        # calculating residual
        r = y - stump.predict(X)
        # calculating loss
        loss = np.mean(r**2)
        hist.append(loss)
        for i in range(1,num_itr):
            # fitting data 
            stump = DecisionTreeRegressor(max_depth=1)
            stump.fit(X,r)
            self.weak_learners.append(stump)
            # calculating residual
            r = r - alpha*stump.predict(X)
            # calculating loss
            loss = np.mean(r**2)
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
        for i,h in enumerate(self.weak_learners):
            if i == 0:
                y = h.predict(X)
                continue
            y += self.alpha*h.predict(X)

        return y