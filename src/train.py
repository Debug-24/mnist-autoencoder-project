from utils import load_mnist_data, check_data_specs
from model import Autoencoder


def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_mnist_data()

    # 2. Verify Data
    check_data_specs(X_train)

    print("\n--- Training Initialization ---")
    print(f"Training on {X_train.shape[0]} images.")
    print("Model initialized. Starting training loop...")

    # 3. Create Autoencoder Model
    model = Autoencoder(
        input_size=784,
        hidden_size=128,
        latent_size=32,
        lr=0.01
    )

    # 4. Training Settings
    epochs = 10
    batch_size = 128

    # 5. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i + batch_size]

            loss = model.train_step(X_batch)

            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    print("\nTraining finished!")


if __name__ == "__main__":
    main()