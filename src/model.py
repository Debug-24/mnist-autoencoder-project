import numpy as np


class Autoencoder:
    def __init__(self, input_size=784, hidden_size=128, latent_size=32, lr=0.01):
        self.lr = lr

        # Encoder weights: original image size -> hidden layer -> compressed layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, latent_size) * 0.01
        self.b2 = np.zeros((1, latent_size))

        # Decoder weights: compressed layer -> hidden layer -> reconstructed image
        self.W3 = np.random.randn(latent_size, hidden_size) * 0.01
        self.b3 = np.zeros((1, hidden_size))

        self.W4 = np.random.randn(hidden_size, input_size) * 0.01
        self.b4 = np.zeros((1, input_size))

    def sigmoid(self, values):
        return 1 / (1 + np.exp(-values))

    def sigmoid_derivative(self, activated_values):
        return activated_values * (1 - activated_values)

    def forward(self, X):
        # Encoder part
        self.encoder_input = X @ self.W1 + self.b1
        self.encoder_hidden = self.sigmoid(self.encoder_input)

        self.latent_input = self.encoder_hidden @ self.W2 + self.b2
        self.latent_vector = self.sigmoid(self.latent_input)

        # Decoder part
        self.decoder_input = self.latent_vector @ self.W3 + self.b3
        self.decoder_hidden = self.sigmoid(self.decoder_input)

        self.reconstruction_input = self.decoder_hidden @ self.W4 + self.b4
        self.reconstructed_output = self.sigmoid(self.reconstruction_input)

        return self.reconstructed_output

    def compute_loss(self, original, reconstructed):
        # Mean Squared Error between original image and reconstructed image
        return np.mean((original - reconstructed) ** 2)

    def backward(self, X):
        batch_count = X.shape[0]

        # Error at output layer
        output_error = (self.reconstructed_output - X) * self.sigmoid_derivative(
            self.reconstructed_output
        )

        dW4 = self.decoder_hidden.T @ output_error / batch_count
        db4 = np.sum(output_error, axis=0, keepdims=True) / batch_count

        # Backpropagate through decoder hidden layer
        decoder_hidden_error = output_error @ self.W4.T
        decoder_hidden_delta = decoder_hidden_error * self.sigmoid_derivative(
            self.decoder_hidden
        )

        dW3 = self.latent_vector.T @ decoder_hidden_delta / batch_count
        db3 = np.sum(decoder_hidden_delta, axis=0, keepdims=True) / batch_count

        # Backpropagate through latent compressed layer
        latent_error = decoder_hidden_delta @ self.W3.T
        latent_delta = latent_error * self.sigmoid_derivative(self.latent_vector)

        dW2 = self.encoder_hidden.T @ latent_delta / batch_count
        db2 = np.sum(latent_delta, axis=0, keepdims=True) / batch_count

        # Backpropagate through encoder hidden layer
        encoder_hidden_error = latent_delta @ self.W2.T
        encoder_hidden_delta = encoder_hidden_error * self.sigmoid_derivative(
            self.encoder_hidden
        )

        dW1 = X.T @ encoder_hidden_delta / batch_count
        db1 = np.sum(encoder_hidden_delta, axis=0, keepdims=True) / batch_count

        # Gradient descent weight updates
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X):
        reconstructed = self.forward(X)
        loss = self.compute_loss(X, reconstructed)
        self.backward(X)
        return loss