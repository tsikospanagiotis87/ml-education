import numpy as np
from tqdm import tqdm

class Regression_model():
    def __init__(self, learning_rate= 0.01, max_itter= 100, bias= True, method= 'gd'):
        self.learning_rate = learning_rate
        self.max_itter = max_itter
        self.bias = bias
        self.method = method

    def cost_function(self, X, y, w):
        res = X @ w - y
        return np.dot(res, res) / (2 * len(y))

    def train(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        if self.method == 'gd':
            w = np.zeros(X.shape[1])
            w_list = []
            cost_list = []
            for itter in tqdm(range(self.max_itter)):
                gradient_w = X.T @ (X @ w - y)
                w -= self.learning_rate * gradient_w / len(y)
                w_list.append(w)
                cost = self.cost_function(X, y, w)
                cost_list.append(cost)
            idx_min_cost = cost_list.index(min(cost_list))
            self.w_trained = w_list[idx_min_cost]
            return self.w_trained
        else:
            XTX = X.T @ X
            XTX_inv = np.linalg.inv(XTX)
            self.w_trained = XTX_inv @ X.T @ y
            return self.w_trained

    def predict(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.w_trained

    def rmse(self, y, y_pred):
        res = y_pred - y
        sq_res = np.dot(res, res)
        rmse = np.sqrt(sq_res / len(y))
        return rmse