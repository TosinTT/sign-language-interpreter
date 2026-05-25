"""
Transfer Learning for Image Classification with TensorFlow 2+

This script performs transfer learning on a pre-trained MobileNetV2 model to classify 
a new set of images. Place your images in subfolders inside the image_dir directory,
with each subfolder representing a class.

Example usage:
python transfer_learning.py --image_dir=C:/Users/ZBOOK STUDIO G5/Documents/sign-language-alphabet-recognizer-master/sign-language-alphabet-recognizer-master/dataset
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import time
import pathlib

# Suppress warnings
tf.get_logger().setLevel('ERROR')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='C:\\Users\\Administrator\\Desktop\\asl_alphabet_train',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='saved_model',
        help='Where to save the trained model.'
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='labels.txt',
        help='Where to save the trained model\'s labels.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='How many training epochs to run before ending.'
    )
    parser.add_argument(
        '--img_height',
        type=int,
        default=224,
        help='Input image height dimension.'
    )
    parser.add_argument(
        '--img_width',
        type=int,
        default=224,
        help='Input image width dimension.'
    )
    parser.add_argument(
        '--data_augmentation',
        action='store_true',
        default=False,
        help='Whether to use data augmentation.'
    )
    parser.add_argument(
        '--fine_tune_layers',
        type=int,
        default=0,
        help='Number of layers to fine-tune from the base model.'
    )
    return parser.parse_args()

def create_data_generators(image_dir, img_height, img_width, batch_size, validation_split):
    """Create train, validation, and test data generators"""
    print(f"Loading data from {image_dir}")
    
    # Check if the directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        sys.exit(1)
    
    # Use data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if FLAGS.data_augmentation else ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Get list of classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Found {len(class_names)} classes: {class_names}")
    
    return train_generator, validation_generator, class_names

def create_model(num_classes, img_height, img_width, fine_tune_layers=0):
    """Create and compile the transfer learning model"""
    print("Creating model with MobileNetV2 base...")
    
    # Create the base model from the pre-trained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', 
                           include_top=False, 
                           input_shape=(img_height, img_width, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Fine-tune the specified number of layers if requested
    if fine_tune_layers > 0:
        # Ensure we don't exceed the number of layers in the model
        fine_tune_layers = min(fine_tune_layers, len(base_model.layers))
        
        # Unfreeze the last N layers
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True
        
        print(f"Fine-tuning the last {fine_tune_layers} layers of the base model.")
    
    # Create a new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_model_and_labels(model, class_names, output_model, output_labels):
    """Save the trained model and labels"""
    # Add .keras extension if not already present
    if not output_model.endswith('.keras') and not output_model.endswith('.h5'):
        output_model = output_model + '.keras'
        
    # Save the model
    model.save(output_model)
    print(f"Model saved to {output_model}")
    
    # Save the labels
    with open(output_labels, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"Labels saved to {output_labels}")

def plot_training_results(history, epochs):
    """Plot training and validation accuracy/loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('training_results.png')
    print("Training results plot saved to training_results.png")

def main():
    # Validate the image directory
    if not FLAGS.image_dir:
        print("Please specify the image directory with --image_dir")
        return 1
    
    # Create data generators
    validation_split = (FLAGS.validation_percentage / 100.0)
    train_generator, validation_generator, class_names = create_data_generators(
        FLAGS.image_dir, 
        FLAGS.img_height, 
        FLAGS.img_width, 
        FLAGS.batch_size, 
        validation_split
    )
    
    # Create and compile the model
    model = create_model(
        num_classes=len(class_names), 
        img_height=FLAGS.img_height, 
        img_width=FLAGS.img_width,
        fine_tune_layers=FLAGS.fine_tune_layers
    )
    
    # Print model summary
    print(model.summary())
    
    # Train the model
    print(f"Training for {FLAGS.epochs} epochs...")
    start_time = time.time()
    
    # Use TensorBoard callback for logging
    log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // FLAGS.batch_size,
        epochs=FLAGS.epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // FLAGS.batch_size,
        callbacks=[tensorboard_callback, early_stopping]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    # Plot training results
    plot_training_results(history, len(history.history['accuracy']))
    
    # Save the model and labels
    save_model_and_labels(model, class_names, FLAGS.output_model, FLAGS.output_labels)
    
    return 0

if __name__ == '__main__':
    FLAGS = parse_arguments()
    main()