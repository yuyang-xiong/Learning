import numpy as np
from ucimlrepo import fetch_ucirepo

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = y.shape[0]
        self.d = X.shape[1]

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.mean(np.square(y_pred - self.y))
        return mse

    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        d_w = (2.0 / self.n) * self.X.T @ (y_pred - self.y)
        d_b = (2.0 / self.n) * np.sum(y_pred - self.y)
        return d_w, d_b


class GradientDescent:
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, loss_func):
        w = np.zeros(loss_func.d)
        b = 0.0
        for epoch in range(self.epochs):
            d_w, d_b = loss_func.compute_gradient(w, b)
            w = w - self.lr * d_w
            b = b - self.lr * d_b
            if epoch % 1000 == 0:
                loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f}")

        return w, b

if __name__ == "__main__":
    abalone = fetch_ucirepo(id=1)
    print(type(abalone))
    print(abalone)

    X_raw = abalone.data.features.values[:, 1:].astype(float)
    print('X_raw')
    print(type(X_raw))
    print(X_raw.shape, X_raw.dtype)
    print(X_raw)

    y_raw = abalone.data.targets.values.ravel()
    print('y_raw')
    print(type(y_raw))
    print(y_raw.shape, y_raw.dtype)
    print(y_raw)

    print("Dataset name:", abalone.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
    loss = LinearRegression(X, y_raw)
    optimizer = GradientDescent(learning_rate=0.01, epochs=10000)
    w, b = optimizer.optimize(loss)

    print("=== Training Finished ===")
    print("Optimal weight w:", w)
    print("Optimal bias b:", b)
    final_loss = loss.compute_loss(w, b)
    print("Final Loss:", final_loss)
