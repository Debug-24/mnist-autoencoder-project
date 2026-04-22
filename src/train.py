from utils import load_mnist_data, check_data_specs
# from model import Autoencoder  

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # 2. Verify Data 
    check_data_specs(X_train)
    
    print("\n--- Training Initialization ---")
    print(f"Training on {X_train.shape[0]} images.")
    print("Model initialized. Ready for training loop...")
    
    # 3. Training Loop will go here once model.py is ready
    # model = Autoencoder(input_dim=784, hidden_dim=32)
    # model.train(X_train)

if __name__ == "__main__":
    main()