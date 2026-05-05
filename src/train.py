from utils import load_mnist_data, check_data_specs
from model import Autoencoder
import numpy as np


def main():
    # load data
    X_train, X_test, y_train, y_test = load_mnist_data()

    # use only digits 0-8 for training
    mask_train = y_train != 9
    X_train = X_train[mask_train]

    # check data
    check_data_specs(X_train)

    print("\n--- start training ---")
    print(f"using digits 0-8 only: {X_train.shape[0]} samples")

    # model
    model = Autoencoder(
        input_size=784,
        hidden_size=64,
        latent_size=16,
        lr=0.01
    )

    epochs = 5
    batch_size = 128

    # training loop
    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for i in range(0, X_train.shape[0], batch_size):
            batch = X_train[i:i + batch_size]

            loss = model.train_step(batch)

            total_loss += loss
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    print("\ntraining done!")

    # anomaly test: digit 9
    mask_normal = y_test != 9
    mask_anomaly = y_test == 9

    X_normal = X_test[mask_normal]
    X_anomaly = X_test[mask_anomaly]

    # reconstruction errors
    err_train = model.reconstruction_error(X_train)
    err_normal = model.reconstruction_error(X_normal)
    err_anomaly = model.reconstruction_error(X_anomaly)

    # threshold from training data
    threshold = np.percentile(err_train, 90)

    normal_flag = np.mean(err_normal > threshold) * 100
    anomaly_flag = np.mean(err_anomaly > threshold) * 100

    print("\n--- anomaly result ---")
    print(f"normal avg error: {np.mean(err_normal):.6f}")
    print(f"digit 9 avg error: {np.mean(err_anomaly):.6f}")
    print(f"threshold: {threshold:.6f}")
    print(f"normal flagged: {normal_flag:.2f}%")
    print(f"digit 9 flagged: {anomaly_flag:.2f}%")

    # check average error for each digit
    print("\n--- error by digit ---")
    for digit in range(10):
        digit_mask = y_test == digit
        digit_error = model.reconstruction_error(X_test[digit_mask])
        print(f"digit {digit} avg error: {np.mean(digit_error):.6f}")


if __name__ == "__main__":
    main()