"""
Real-Time Sign Language Detection with TensorFlow 2+

This script uses a webcam to capture sign language gestures and
classifies them in real-time using a trained TensorFlow model.
Only displays recognized signs.
"""

import sys
import os
import numpy as np
import cv2
import tensorflow as tf

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SignLanguageDetector:
    def __init__(self, model_path, labels_path):
        """Initialize the detector with model and labels"""
        print("Initializing Sign Language Detector...")
        
        # Load the model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load labels
        print(f"Loading labels from: {labels_path}")
        self.load_labels(labels_path)
        
        # Get input shape expected by the model
        self.img_height = self.model.input_shape[1]
        self.img_width = self.model.input_shape[2]
        print(f"Model expects input shape: {self.img_height}x{self.img_width}")
        
    def load_labels(self, labels_path):
        """Load class labels from a text file"""
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.labels)} classes: {self.labels}")
        
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        # Resize image to match model's expected input
        resized = cv2.resize(img, (self.img_height, self.img_width))
        
        # Convert to RGB (from BGR)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
        
    def predict(self, img):
        """Make prediction on image"""
        # Preprocess the image
        processed_img = self.preprocess_image(img)
        
        # Run prediction
        predictions = self.model.predict(processed_img, verbose=0)
        
        # Get the highest probability class
        top_idx = np.argmax(predictions[0])
        top_label = self.labels[top_idx]
        top_score = float(predictions[0][top_idx])
        
        return top_label, top_score

def main():
    # Set paths for model and labels
    model_path = "saved_model.keras"  # Update with your model path
    labels_path = "labels.txt"        # Update with your labels path
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return 1
    if not os.path.exists(labels_path):
        print(f"Error: Labels file '{labels_path}' not found.")
        return 1
    
    # Initialize detector
    detector = SignLanguageDetector(model_path, labels_path)
    
    # Initialize webcam
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 1
    
    # Set up variables for prediction and display
    res, score = '', 0.0
    i = 0  # Frame counter for prediction frequency
    mem = ''  # Remember last prediction
    consecutive = 0  # Count consecutive same predictions
    sequence = ''  # Text sequence of recognized signs
    
    # Minimum confidence threshold
    CONFIDENCE_THRESHOLD = 0.5  # Adjust this value based on your model's performance
    
    print("Ready! Press ESC to quit.")
    
    while True:
        # Read frame from webcam
        ret, img = cap.read()
        
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break
        
        # Flip image horizontally for a more intuitive mirror effect
        img = cv2.flip(img, 1)
        
        # Define region of interest (ROI) for hand gestures - LARGER BOX
        x1, y1, x2, y2 = 50, 50, 400, 400  # Increased box size
        img_cropped = img[y1:y2, x1:x2]
        
        # Check if a hand is present in the ROI using simple skin detection
        def detect_hand(roi):
            # Convert to HSV color space for better skin detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask for skin color
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin pixels in ROI
            skin_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            skin_percentage = (skin_pixels / total_pixels) * 100
            
            # Return True if enough skin pixels are detected (adjust threshold as needed)
            return skin_percentage > 5  # Adjust this threshold based on testing
        
        hand_present = detect_hand(img_cropped)
        
        # Make prediction every 5 frames to reduce CPU usage and only if hand is present
        if i % 5 == 0 and hand_present:
            res_tmp, score = detector.predict(img_cropped)
            
            # Only update the result if it's not "nothing" and has sufficient confidence
            if res_tmp != 'nothing' and score > CONFIDENCE_THRESHOLD:
                res = res_tmp
            else:
                res = ''  # Clear the result if it's "nothing" or low confidence
        elif not hand_present:
            res = ''  # Clear the result if no hand is detected
            
            # Track consecutive identical predictions
            if mem == res and res != '':
                consecutive += 1
            else:
                consecutive = 0
                
            # After seeing the same prediction 2 times, add to sequence
            if consecutive == 2 and res != '':
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    if sequence:
                        sequence = sequence[:-1]
                else:
                    sequence += res
                consecutive = 0
                
            mem = res
        
        i += 1
        
        # Draw rectangle around ROI - change color based on hand detection
        rectangle_color = (0, 255, 0) if hand_present else (255, 0, 0)  # Green if hand detected, red otherwise
        cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, 2)
        
        # Display the prediction only if something is recognized
        if res:
            cv2.putText(img, f"{res.upper()}", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
            cv2.putText(img, f"(score = {score:.5f})", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        
        # Show the image
        cv2.imshow("Sign Language Detection", img)
        
        # Create a black image for displaying the sequence
        img_sequence = np.zeros((200, 1200, 3), np.uint8)
        cv2.putText(img_sequence, f"{sequence.upper()}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Sequence', img_sequence)
        
        # Check for ESC key press
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    # Clean up
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    main()