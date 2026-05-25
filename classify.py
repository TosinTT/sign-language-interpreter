"""
Sign Language Classification Script for TensorFlow 2+

This script loads a trained model and uses it to classify a sign language image.

Usage:
python classify.py path/to/image.jpg
"""

import tensorflow as tf
import numpy as np
import sys
import os
import pathlib
from PIL import Image

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_image(image_path, img_height=224, img_width=224):
    """Load and prepare an image for classification"""
    print(f"Loading image: {image_path}")
    
    # Read the image file
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(img_height, img_width)
    )
    
    # Convert the image to a numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def load_labels(label_path):
    """Load labels from a text file"""
    print(f"Loading labels from: {label_path}")
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Please provide an image path")
        print("Usage: python classify.py path/to/image.jpg")
        return 1
    
    # Get image path from command line argument
    image_path = sys.argv[1]
    
    # Set model and label paths (update these to match your saved files)
    model_path = "saved_model.keras"  # or your actual model path
    label_path = "labels.txt"        # or your actual labels path
    
    # Check if paths exist
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' not found.")
        return 1
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' not found.")
        return 1
    if not os.path.exists(label_path):
        print(f"Error: Label path '{label_path}' not found.")
        return 1
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load the labels
    class_names = load_labels(label_path)
    
    # Load and prepare the image
    img_array = load_and_prepare_image(image_path)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    
    # Print all predictions sorted by confidence
    print("\nPrediction Results:")
    print("-" * 30)
    
    # Sort indices by prediction values (highest first)
    top_indices = np.argsort(predictions[0])[::-1]
    
    for i, idx in enumerate(top_indices):
        class_name = class_names[idx]
        confidence = predictions[0][idx]
        print(f"{class_name}: {confidence:.5f} ({confidence*100:.2f}%)")
        
        # Only show top 5 predictions
        if i >= 4:
            break
    
    print("\nTop prediction:", class_names[predicted_class_index])
    return 0

if __name__ == "__main__":
    main()