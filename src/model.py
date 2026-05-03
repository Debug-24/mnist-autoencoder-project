import numpy as np


class Autoencoder:
    def __init__(self, input_size=784, hidden_size=128, latent_size=32, lr=0.01):
        self.lr = lr

        # encoder: reduce image size step by step
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, latent_size) * 0.01
        self.b2 = np.zeros((1, latent_size))

        # decoder: rebuild image back to original size
        self.W3 = np.random.randn(latent_size, hidden_size) * 0.01
        self.b3 = np.zeros((1, hidden_size))

        self.W4 = np.random.randn(hidden_size, input_size) * 0.01
        self.b4 = np.zeros((1, input_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        # encoder forward
        self.h1 = self.sigmoid(X @ self.W1 + self.b1)
        self.z = self.sigmoid(self.h1 @ self.W2 + self.b2)

        # decoder forward
        self.h2 = self.sigmoid(self.z @ self.W3 + self.b3)
        self.out = self.sigmoid(self.h2 @ self.W4 + self.b4)

        return self.out

    def compute_loss(self, X, output):
        # difference between original and reconstructed image
        return np.mean((X - output) ** 2)

    def backward(self, X):
        n = X.shape[0]

        # output layer gradient
        d_out = (self.out - X) * self.sigmoid_derivative(self.out)

        dW4 = self.h2.T @ d_out / n
        db4 = np.sum(d_out, axis=0, keepdims=True) / n

        # decoder hidden
        d_h2 = (d_out @ self.W4.T) * self.sigmoid_derivative(self.h2)

        dW3 = self.z.T @ d_h2 / n
        db3 = np.sum(d_h2, axis=0, keepdims=True) / n

        # latent layer
        d_z = (d_h2 @ self.W3.T) * self.sigmoid_derivative(self.z)

        dW2 = self.h1.T @ d_z / n
        db2 = np.sum(d_z, axis=0, keepdims=True) / n

        # encoder hidden
        d_h1 = (d_z @ self.W2.T) * self.sigmoid_derivative(self.h1)

        dW1 = X.T @ d_h1 / n
        db1 = np.sum(d_h1, axis=0, keepdims=True) / n

        # update weights
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # used for anomaly detection
    def reconstruction_error(self, X):
        recon = self.forward(X)
        return np.mean((X - recon) ** 2, axis=1)

    def train_step(self, X):
        out = self.forward(X)
        loss = self.compute_loss(X, out)
        self.backward(X)
        return loss