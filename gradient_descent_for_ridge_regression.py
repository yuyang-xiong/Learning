import numpy as np
from ucimlrepo import fetch_ucirepo


class LinearRegression:
    def __init__(self, X, y, lambda_=0.0):
        self.X = X   # abalone features
        self.y = y   # abalone age
        self.n = y.shape[0]   # number of samples
        self.d = X.shape[1]   # sample dimension
        self.lambda_ = lambda_   # L2‑penalty coefficient

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.mean(np.square(y_pred - self.y))
        l2_penalty = self.lambda_ * np.sum(np.square(w))
        loss = mse + l2_penalty
        return loss

    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        d_w = (2.0 / self.n) * self.X.T @ (y_pred - self.y) + 2 * self.lambda_ * w
        d_b = (2.0 / self.n) * np.sum(y_pred - self.y)
        return d_w, d_b


class GradientDescent:
    def __init__(self, learning_rate=0.01, n_epochs=2000):
        self.lr = learning_rate   # learning rate for gradient‑descent
        self.n_epochs = n_epochs   # number of training iterations

    def optimize(self, loss_func):
        w = np.zeros(loss_func.d)
        b = 0.0
        for epoch in range(self.n_epochs):
            d_w, d_b = loss_func.compute_gradient(w, b)
            w = w - self.lr * d_w
            b = b - self.lr * d_b

        return w, b


if __name__ == "__main__":
    abalone = fetch_ucirepo(id=1)

    X_raw = abalone.data.features.values[:, 1:].astype(float)   # shape: (4177, 7)
    y_raw = abalone.data.targets.values.ravel()   # shape: (4177,)

    print("Dataset name:", abalone.metadata.name)   # Abalone
    print("Number of samples:", X_raw.shape[0])
    print("Number of features:", X_raw.shape[1])

    lambda_ = 0.05
    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    loss = LinearRegression(X, y_raw, lambda_=lambda_)
    optimizer = GradientDescent(learning_rate=0.01, n_epochs=5000)
    w, b = optimizer.optimize(loss)

    print("=== Training Finished ===")
    print("Optimal weight w:", w)
    print("Optimal bias b:", b)
    print("Final Loss:", loss.compute_loss(w, b))
