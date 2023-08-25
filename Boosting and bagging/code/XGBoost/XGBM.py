import numpy as np

class DescisionStump:
    def __init__(self,g = None,reg = 0) -> None:
        """
        Input
        --------------------------------
        g: ndarray of shape (Num,)
        reg: int
        --------------------------------
        """
        self.g = g
        self.reg = reg
        self.right = None
        self.left = None
        self.threshold = None
        self.right_weight = None
        self.left_weight = None
        self.feature_criteria = None

    def split(self,X,y):
        gain = 0
        G = np.sum(self.g)
        H = self.g.shape[0]
        m = X.shape[1]
        bestGR = None
        bestGL = None
        for k in range(m):
            GL = 0
            HL = 0
            for j in np.argsort(X[:,k]):
                GL += self.g[j]
                HL += 1
                GR = G - GL
                HR = H - HL
                Lsplit = GR**2/(HR + self.reg) + GL**2/(HL + self.reg) - G**2/(H + self.reg)
                if (gain > Lsplit):
                    gain = Lsplit
                    self.feature_criteria = k
                    self.threshold = j
                    self.left, self.right = np.split(X,[j+1])     
                    bestGR = GR
                    bestGL = GL

        self.left_weight = bestGL/(self.left_weight.shape[0] + self.reg)
        self.right_weight = bestGR/(self.right_weight.shape[0] + self.reg)
    
    def fit(self,X,y):
        """
        Input
        --------------------------------
        X: ndarray of shape (Num,features)
        y: ndarray of shape (Num,)
        --------------------------------
        """
        if self.g is None:
            self.g = np.random.randn(X.shape[0])
        self.split(X,y)
    
    def predict(self,X):
        """
        Input
        --------------------------------
        X: ndarray of shape (Num,features)
        --------------------------------

        Output:
        --------------------------------
        y_pred: ndarray of shape (Num,)
        --------------------------------
        """
        right_mask = X[:,self.feature_criteria] > self.threshold
        left_mask = ~right_mask
        y_pred = np.empty(X.shape[0]) 
        y_pred[right_mask] = self.right_weight
        y_pred[left_mask] = self.left_weight

        return y_pred    
        

class XGBM:
    def __init__(self) -> None:
        self.hypothesis = None
        self.alpha = None

    def train(self,X,y,reg,alpha,num_itr):
        """
        Input
        --------------------------------
        X: ndarray of shape (Num,features)
        y: ndarray of shape (num,)
        num_itteration: int
        --------------------------------
        """
        # initialization
        self.hypothesis = []
        self.alpha = alpha
        # training
        stump = DescisionStump(reg=reg)
        stump.fit(X,y)
        # updating g
        g = stump.predict(X) - y
        for i in range(1,num_itr):
            # Training data
            stump = DescisionStump(reg=reg,g=g)
            stump.fit(X,y)
            self.hypothesis.append(stump)
            # updating g
            g += alpha*stump.predict(X)
        
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
        
        for i,h in enumerate(self.hypothesis):
            if i == 0:
                y = h.predict(X)
                continue
            y += self.alpha*h.predict(X)

        return y