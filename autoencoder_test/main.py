import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the images for the autoencoder
x_train_flat = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_flat = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Autoencoder architecture
input_dim = 784  # 28x28 pixels
encoding_dim = 32  # Size of the encoded representation

# Encoder
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_img, decoded)

# Encoder model (to get encoded representations)
encoder = Model(input_img, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train_flat, x_train_flat,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_flat, x_test_flat))

# Function to get encoded representations
def get_encoded(image):
    if image.ndim == 2:
        image = image.reshape(1, -1)  # Reshape if single image
    return encoder.predict(image)

# Function to calculate similarity between two images
def calculate_similarity(img1, img2):
    # Get encoded representations
    encoded1 = get_encoded(img1)
    encoded2 = get_encoded(img2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(encoded1, encoded2)[0][0]
    return similarity

# Example usage: Check similarity between test images
def show_similarity(idx1, idx2):
    img1 = x_test[idx1]
    img2 = x_test[idx2]
    
    similarity = calculate_similarity(img1, img2)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(f'Digit {y_test[idx1]}')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(f'Digit {y_test[idx2]}')
    
    plt.suptitle(f'Similarity: {similarity:.4f}')
    plt.show()

# Compare similar digits (two 5s)
show_similarity(0, 2)  # Change indices to see different comparisons

# Compare different digits (5 and 0)
show_similarity(0, 1)