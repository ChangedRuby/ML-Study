import numpy as np


class PolynomialRegression:
    def __init__(self,X_train, Y_train, q = 1):
        self.X = np.copy(X_train)
        self.N, self.p = X_train.shape
        self.q = q
        for i in range(q):
            if i == 0:
                x = np.ones((self.N,1))
            else:
                x = (X_train)**(i+1)
            self.X = np.concatenate((
                self.X, x
            ),axis=1)
        self.Y = Y_train
        bp=1
    def fit(self):
        # beta = np.linalg.inv(self.X.T@self.X)@self.X.T@self.Y
        self.beta = np.linalg.pinv(self.X)@self.Y
        # beta3 = np.linalg.lstsq(self.X, self.Y)[0]
    def predict(self, X):
        Xt = np.copy(X)
        if len(X.shape)>2:
            z,k,j = X.shape
            for i in range(self.q):
                if i == 0:
                    x = np.ones((z,k,1))
                else:
                    x = (X)**(i+1)
                Xt = np.concatenate((
                    Xt, x
                ),axis=2)
            return Xt@self.beta
        else:
            N,p = X.shape
            for i in range(self.q):
                if i == 0:
                    x = np.ones((N,1))
                else:
                    x = (X)**(i+1)
                Xt = np.concatenate((
                    Xt, x
                ),axis=1)
            return Xt@self.beta
        
        


class LinearRegression:
    def __init__(self,X_train, Y_train,fit_intercept = True,solver = 'OLS'):
        self.X = X_train
        self.N, self.p = X_train.shape
        if fit_intercept:
            self.X = np.hstack((
                np.ones((self.N,1)), self.X
            ))
        self.Y = Y_train
        self.fit_in = fit_intercept
        self.solver = solver
        
    def fit(self):
        self.beta = np.linalg.inv(self.X.T @ self.X)@self.X.T@self.Y
    
    def predict(self, X):
        N,p = X.shape
        if self.fit_in:
            Xt = np.hstack((
                np.ones((N,1)),X
            ))
        else:
            Xt = np.copy(X)
        
        return Xt@self.beta
