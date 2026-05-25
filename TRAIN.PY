"""
Enhanced Transfer Learning for Sign Language Alphabet Classification with TensorFlow 2+

This script performs transfer learning on a pre-trained MobileNetV2 model to classify 
sign language alphabets. It generates comprehensive reports including CSV files, 
visualizations, and training metrics.

Example usage:
python enhanced_transfer_learning.py --image_dir=C:/Users/Administrator/Desktop/Tosin/alphabets
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pathlib
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from collections import Counter
import random

# Suppress warnings
tf.get_logger().setLevel('ERROR')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='C:/Users/Administrator/Desktop/Tosin/alphabets',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='C:/Users/Administrator/Desktop/Tosin',
        help='Where to save all outputs.'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='sign_language_model',
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
        default=20,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=20,
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
        default=20,
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
        default=True,
        help='Whether to use data augmentation.'
    )
    parser.add_argument(
        '--fine_tune_layers',
        type=int,
        default=20,
        help='Number of layers to fine-tune from the base model.'
    )
    return parser.parse_args()

def generate_dataset_csv(image_dir, output_dir):
    """Generate CSV files for train and test datasets"""
    print("Generating dataset CSV files...")
    
    train_data = []
    test_data = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(image_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split images into train and test
        np.random.shuffle(image_files)
        test_split = int(len(image_files) * (FLAGS.testing_percentage / 100))
        
        test_files = image_files[:test_split]
        train_files = image_files[test_split:]
        
        # Add to train data
        for img_file in train_files:
            train_data.append({
                'filename': img_file,
                'filepath': os.path.join(class_path, img_file),
                'class': class_name,
                'split': 'train'
            })
        
        # Add to test data
        for img_file in test_files:
            test_data.append({
                'filename': img_file,
                'filepath': os.path.join(class_path, img_file),
                'class': class_name,
                'split': 'test'
            })
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Save CSV files
    train_csv_path = os.path.join(output_dir, 'train_dataset.csv')
    test_csv_path = os.path.join(output_dir, 'test_dataset.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Train dataset CSV saved to: {train_csv_path}")
    print(f"Test dataset CSV saved to: {test_csv_path}")
    
    return train_df, test_df

def create_alphabet_bar_chart(train_df, test_df, output_dir):
    """Create bar chart for alphabet distribution"""
    print("Creating alphabet distribution bar chart...")
    
    # Count samples per class
    train_counts = train_df['class'].value_counts().sort_index()
    test_counts = test_df['class'].value_counts().sort_index()
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Train distribution
    train_counts.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.7)
    ax1.set_title('Training Set - Alphabet Distribution', fontsize=14)
    ax1.set_xlabel('Alphabet')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    
    # Test distribution
    test_counts.plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.7)
    ax2.set_title('Test Set - Alphabet Distribution', fontsize=14)
    ax2.set_xlabel('Alphabet')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'alphabet_distribution.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Alphabet distribution chart saved to: {chart_path}")

def display_resized_images(train_df, output_dir, img_height, img_width):
    """Display 10 random images after resizing"""
    print("Creating resized images display...")
    
    # Select 10 random images
    sample_images = train_df.sample(10)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(sample_images.iterrows()):
        # Load and resize image
        img = cv2.imread(row['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (img_width, img_height))
        
        # Display image
        axes[i].imshow(img_resized)
        axes[i].set_title(f"Class: {row['class']}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Images After Resizing (224x224)', fontsize=16)
    plt.tight_layout()
    
    resized_path = os.path.join(output_dir, 'resized_images_sample.png')
    plt.savefig(resized_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Resized images sample saved to: {resized_path}")

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

def create_training_table(history, output_dir):
    """Create a comprehensive training table"""
    print("Creating training metrics table...")
    
    # Extract training metrics
    epochs = range(1, len(history.history['accuracy']) + 1)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    
    # Create DataFrame
    training_df = pd.DataFrame({
        'Epoch': epochs,
        'Training_Accuracy': train_acc,
        'Training_Loss': train_loss,
        'Validation_Accuracy': val_acc,
        'Validation_Loss': val_loss
    })
    
    # Save as CSV
    table_path = os.path.join(output_dir, 'training_metrics.csv')
    training_df.to_csv(table_path, index=False)
    
    # Create a visual table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data for display
    display_data = training_df.round(4)
    
    table = ax.table(cellText=display_data.values, colLabels=display_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(display_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Training Metrics by Epoch', fontsize=16, fontweight='bold', pad=20)
    
    table_img_path = os.path.join(output_dir, 'training_metrics_table.png')
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics table saved to: {table_path}")
    print(f"Training metrics table image saved to: {table_img_path}")
    
    return training_df

def evaluate_model_comprehensive(model, validation_generator, class_names, output_dir):
    """Comprehensive model evaluation"""
    print("Performing comprehensive model evaluation...")
    
    # Get predictions
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = validation_generator.classes
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(validation_generator, verbose=0)
    
    # Create classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, output_dict=True)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    
    return test_accuracy, test_loss, report_df

def save_model_and_labels(model, class_names, output_dir, model_name):
    """Save the trained model and labels"""
    model_path = os.path.join(output_dir, f"{model_name}.keras")
    labels_path = os.path.join(output_dir, "labels.txt")
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save the labels
    with open(labels_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"Labels saved to {labels_path}")

def plot_training_results(history, output_dir):
    """Plot training and validation accuracy/loss"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, acc, 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    plt.plot(epochs, loss, 'b--', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r--', label='Val Loss', linewidth=2)
    plt.title('Training Overview')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training results plot saved to: {results_path}")

def create_final_summary(training_df, test_accuracy, test_loss, output_dir, training_time):
    """Create a final summary report"""
    print("Creating final summary report...")
    
    # Get best metrics
    best_train_acc = training_df['Training_Accuracy'].max()
    best_val_acc = training_df['Validation_Accuracy'].max()
    final_train_acc = training_df['Training_Accuracy'].iloc[-1]
    final_val_acc = training_df['Validation_Accuracy'].iloc[-1]
    
    # Create summary data
    summary_data = {
        'Metric': [
            'Best Training Accuracy',
            'Best Validation Accuracy',
            'Final Training Accuracy',
            'Final Validation Accuracy',
            'Test Accuracy',
            'Test Loss',
            'Training Time (seconds)',
            'Total Epochs'
        ],
        'Value': [
            f"{best_train_acc:.4f}",
            f"{best_val_acc:.4f}",
            f"{final_train_acc:.4f}",
            f"{final_val_acc:.4f}",
            f"{test_accuracy:.4f}",
            f"{test_loss:.4f}",
            f"{training_time:.2f}",
            len(training_df)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'training_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create visual summary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Sign Language Model Training Summary', fontsize=18, fontweight='bold', pad=20)
    
    summary_img_path = os.path.join(output_dir, 'training_summary_table.png')
    plt.savefig(summary_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training summary saved to: {summary_path}")
    print(f"Training summary image saved to: {summary_img_path}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Validate the image directory
    if not FLAGS.image_dir:
        print("Please specify the image directory with --image_dir")
        return 1
    
    # Generate dataset CSV files
    train_df, test_df = generate_dataset_csv(FLAGS.image_dir, FLAGS.output_dir)
    
    # Create alphabet distribution bar chart
    create_alphabet_bar_chart(train_df, test_df, FLAGS.output_dir)
    
    # Display resized images
    display_resized_images(train_df, FLAGS.output_dir, FLAGS.img_height, FLAGS.img_width)
    
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
    
    # Use callbacks
    log_dir = os.path.join(FLAGS.output_dir, "logs/fit/" + time.strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
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
    
    # Create training metrics table
    training_df = create_training_table(history, FLAGS.output_dir)
    
    # Comprehensive model evaluation
    test_accuracy, test_loss, report_df = evaluate_model_comprehensive(
        model, validation_generator, class_names, FLAGS.output_dir
    )
    
    # Plot training results
    plot_training_results(history, FLAGS.output_dir)
    
    # Save the model and labels
    save_model_and_labels(model, class_names, FLAGS.output_dir, FLAGS.output_model)
    
    # Create final summary
    create_final_summary(training_df, test_accuracy, test_loss, FLAGS.output_dir, training_time)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"All outputs saved to: {FLAGS.output_dir}")
    print("="*50)
    
    return 0

if __name__ == '__main__':
    FLAGS = parse_arguments()
    main()