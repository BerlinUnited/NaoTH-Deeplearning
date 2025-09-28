import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import os, sys
import mlflow.tensorflow
import numpy as np

tools_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(tools_path)
from tools.mflow_helper import set_tracking_url

set_tracking_url()


# Enable MLFlow autologging for TensorFlow
mlflow.tensorflow.autolog()

def create_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    """Create a simple CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def prepare_mnist_data():
    """Load and prepare MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def main():
    # Set up MLFlow experiment
    mlflow.set_experiment("Tests")
    
    # Start MLFlow run
    with mlflow.start_run():
        # Prepare data
        (x_train, y_train), (x_test, y_test) = prepare_mnist_data()
        
        # Create model
        model = create_simple_cnn()
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Log parameters manually (optional)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("batch_size", 128)
        mlflow.log_param("epochs", 5)
        
        # Train model
        print("Training CNN...")
        history = model.fit(x_train, y_train,
                          batch_size=128,
                          epochs=5,
                          validation_split=0.2,
                          verbose=1)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Log additional metrics manually
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        # Save the model
        model.save("mnist_cnn_model.h5")
        mlflow.log_artifact("mnist_cnn_model.h5")
        
        print("Training completed! Check MLFlow UI for results.")

if __name__ == "__main__":
    main()