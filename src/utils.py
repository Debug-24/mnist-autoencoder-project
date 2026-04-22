import kagglehub
import os
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_data():
    print("Downloading/Locating dataset...")
    path = kagglehub.dataset_download("hichamachahboun/mnist-handwritten-digits")
    
    # Load the .npy files
    X_train = np.load(os.path.join(path, 'train_images.npy'))
    y_train = np.load(os.path.join(path, 'train_labels.npy'))
    X_test = np.load(os.path.join(path, 'test_images.npy'))
    y_test = np.load(os.path.join(path, 'test_labels.npy'))
    
    # Preprocess: Flatten and Normalize
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
    
    return X_train, X_test, y_train, y_test

def check_data_specs(X_train):
    print("\n--- Data Specification Check ---")
    print(f"Shape: {X_train.shape}") 
    print(f"Data Type: {X_train.dtype}") 
    print(f"Min value: {X_train.min()}") 
    print(f"Max value: {X_train.max()}")

def visualize_sample(X_train, index=0):
    sample = X_train[index]
    image = sample.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Sample at index {index}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_data()
    check_data_specs(X_train)
    print("Success! Data is loaded and verified.")