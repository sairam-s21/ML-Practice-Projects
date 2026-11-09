import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        # Initializing hyperparameters
        self.learning_rate=learning_rate
        self.n_iters=n_iters
        self.w=None
        self.b=None
    # Function to train model
    def fit(self, X, y):
        n_rows, n_cols = X.shape
        # Initializing parameters
        self.w=np.zeros(n_cols)
        self.b=0.0
        # Gradient Descent
        for _ in  range(self.n_iters):
            y_pred=self.predict(X)
            dw=np.dot(X.T,y_pred-y)/n_rows
            db=np.sum(y_pred-y)/n_rows
            # Update Equations
            self.w-=self.learning_rate*dw
            self.b-=self.learning_rate*db

    # Function to predict output for given input features
    def predict(self, X):
        return np.dot(X,self.w)+self.b

    # Calculates Mean Squared Error
    def mse(self,y_pred,y):
        return np.sum((y_pred-y)**2)/len(y)

    # Calculates Root Mean Squared Error
    def rmse(self,y_pred,y):
        return np.sqrt(self.mse(y_pred,y))

    # Calculates Mean Absolute Error
    def mae(self,y_pred,y):
        return np.sum(np.abs(y_pred-y))/len(y)

    # Calculates R^2 score
    def r2_score(self,y_pred,y):
        ss_res=np.sum((y-y_pred)**2)
        ss_tot=np.sum((y-np.mean(y))**2)
        return 1-ss_res/ss_tot