"""
Real-Time Sign Language Detection with TensorFlow 2+ and ESspeak Voice Output
Simplified for Raspberry Pi with only ESspeak audio output

This script uses a webcam to capture sign language gestures and
classifies them in real-time using a trained TensorFlow model.
Only displays and speaks recognized signs when a hand is detected.
"""

import sys
import os
import numpy as np
import cv2
import tensorflow as tf
import subprocess
import threading
import time

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
        
        # Create a lock for speech engine to prevent overlapping speech
        self.speech_lock = threading.Lock()
        
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
    
    def speak_text(self, text):
        """Speak the given text using espeak"""
        if not text:
            return
            
        # Use threading to avoid blocking the main program flow
        def speak_worker():
            with self.speech_lock:
                try:
                    subprocess.run(["espeak", "-s", "150", "-v", "en", text], 
                                 capture_output=True, check=False)
                except Exception as e:
                    print(f"ESspeak error: {e}")
                    
        thread = threading.Thread(target=speak_worker)
        thread.daemon = True
        thread.start()

def detect_hand(roi):
    """Detect if a hand is present in the ROI using skin detection"""
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
    
    # Return True if enough skin pixels are detected
    return skin_percentage > 5  # Adjust this threshold based on testing

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
    last_spoken_sign = ''  # Track the last sign that was spoken
    
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
        
        # Define region of interest (ROI) for hand gestures
        x1, y1, x2, y2 = 50, 50, 400, 400
        
        # Ensure ROI is within the image bounds
        height, width = img.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Extract ROI if dimensions are valid
        if x1 < x2 and y1 < y2:
            img_cropped = img[y1:y2, x1:x2]
            
            # Check if a hand is present in the ROI
            hand_present = detect_hand(img_cropped)
            
            # Make prediction only if hand is present
            if hand_present:
                # Make prediction every 5 frames to reduce CPU usage
                if i % 5 == 0:
                    res_tmp, score = detector.predict(img_cropped)
                    
                    # Only update the result if it has sufficient confidence
                    if score > CONFIDENCE_THRESHOLD:
                        res = res_tmp
                        
                        # Track consecutive identical predictions
                        if mem == res:
                            consecutive += 1
                        else:
                            consecutive = 0
                            
                        # After seeing the same prediction 2 times, add to sequence and speak
                        if consecutive == 2:
                            if res != last_spoken_sign:  # Only speak if this is a new sign
                                if res == 'space':
                                    sequence += ' '
                                    detector.speak_text("space")
                                elif res == 'del':
                                    # Delete characters continuously while hand is present
                                    if sequence:
                                        sequence = sequence[:-1]  # Remove last character
                                    detector.speak_text("delete")
                                elif res.lower() not in ['nothing', 'null', 'none', 'background']:
                                    # Only add to sequence if it's not a "nothing" class
                                    sequence += res
                                    # Speak the recognized alphabet using espeak
                                    detector.speak_text(res)
                                
                                last_spoken_sign = res  # Update last spoken sign
                        
                        # Continue deleting while del gesture is held
                        if res == 'del' and consecutive > 2:
                            # Delete every 10 frames (adjust for deletion speed)
                            if i % 10 == 0 and sequence:
                                sequence = sequence[:-1]
                    else:
                        res = ''  # Clear the result if it has low confidence
                        
                    mem = res
            else:
                res = ''  # Clear the result if no hand is detected
                consecutive = 0  # Reset consecutive counter
                last_spoken_sign = ''  # Reset last spoken sign
        else:
            hand_present = False
            res = ''
            
        i += 1
        
        # Draw rectangle around ROI - change color based on hand detection
        rectangle_color = (0, 255, 0) if hand_present else (255, 0, 0)  # Green if hand detected, red otherwise
        cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, 2)
        
        # Display the prediction in the OpenCV window (including "nothing" for debugging)
        if res:
            # Display sign in large, clear text
            cv2.putText(img, f"{res.upper()}", (x2 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(img, f"Score: {score:.2f}", (x2 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
        # Display detection status
        status_text = "HAND DETECTED" if hand_present else "NO HAND DETECTED"
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rectangle_color, 2)
        
        # Display instructions
        cv2.putText(img, "Place hand in green box", (width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "Press ESC to quit", (width - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show the main window
        cv2.imshow("Sign Language Detection", img)
        
        # Create a separate window for the sequence display (filtered text)
        img_sequence = np.zeros((100, width, 3), np.uint8)
        cv2.putText(img_sequence, "Detected Sequence:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        # Display the sequence (this won't show "nothing" signs)
        display_sequence = sequence.upper() if sequence else "..."
        cv2.putText(img_sequence, display_sequence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow('Detected Sequence', img_sequence)
        
        # Check for ESC key press
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    # Clean up
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)