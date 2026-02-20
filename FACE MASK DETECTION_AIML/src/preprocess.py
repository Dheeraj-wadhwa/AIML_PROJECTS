import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(dataset_dir, target_size=(224, 224), batch_size=32):
    """
    Creates and returns training and validation data generators.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2  # 20% validation split
    )

    # Only rescaling for validation/testing
    # Note: We use the same generator logic for simplicity, relying on 'subset'
    
    print("Preparing Training Generator...")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    print("Preparing Validation Generator...")
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    # Test the generators
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    try:
        train_gen, val_gen = get_data_generators(base_dir)
        print(f"Classes: {train_gen.class_indices}")
    except Exception as e:
        print(f"Error: {e}")
