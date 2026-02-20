import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from preprocess import get_data_generators
import os
import matplotlib.pyplot as plt

def build_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # 2 classes: with_mask, without_mask
    ])
    
    return model

def train_model(dataset_dir, models_dir, epochs=20, batch_size=32):
    train_gen, val_gen = get_data_generators(dataset_dir, batch_size=batch_size)
    
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen
    )
    
    # Save model
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, 'mask_detector.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return history

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset')
    models_path = os.path.join(base_dir, 'models')
    
    try:
        history = train_model(dataset_path, models_path, epochs=10) # 10 epochs for demo
        plot_history(history)
    except Exception as e:
        print(f"An error occurred: {e}")
