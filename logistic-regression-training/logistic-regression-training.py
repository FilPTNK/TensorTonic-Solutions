import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    m, n = X.shape #m: sample size, n: feature size
    w = np.zeros(n)
    b = 0
    
    for i in range(steps):
        p = sigmoid(X@w + b)
        
        dw = 1/m * np.dot(X.T, p-y)
        db = 1/m * np.sum(p-y)

        w -= lr*dw
        b -= lr*db
    return w,b