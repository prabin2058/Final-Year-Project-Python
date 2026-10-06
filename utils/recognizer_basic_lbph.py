"""
Basic LBPH Face Recognizer - Simple and Working
Based on OpenCV's built-in LBPHFaceRecognizer
Uses user's preprocessing code for better accuracy
"""

import cv2
import numpy as np
import os
import pickle
from utils.preprocess import preprocess_image, preprocess_image_advanced, preprocess_image_simple_clahe


class BasicLBPHRecognizer:
    """
    Simple LBPH Face Recognizer using OpenCV's built-in implementation.
    Now with enhanced CLAHE preprocessing for better accuracy.
    """
    
    def __init__(self, preprocessing_method='standard'):
        """
        Initialize the LBPH recognizer.
        
        Args:
            preprocessing_method: Method for preprocessing
                - 'standard' (default): Original histogram equalization (stable, proven)
                - 'clahe': CLAHE with split-image processing (enhanced contrast)
                - 'advanced': Multi-step CLAHE with normalization (maximum enhancement)
                - 'simple': Simple CLAHE without splitting (faster)
        """
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.labels_to_name = {}  # Maps label ID to person info
        self.is_trained = False
        self.preprocessing_method = preprocessing_method
        print(f"🔧 Preprocessing method: {preprocessing_method.upper()}")
    
    def train_from_directory(self, faces_dir):
        """
        Train the model from a directory structure.
        Expected structure: faces_dir/PersonName_RollNo/image1.jpg, image2.jpg, ...
        
        Args:
            faces_dir: Path to faces directory
        """
        print(f"🎓 Training Basic LBPH from: {faces_dir}")
        
        if not os.path.exists(faces_dir):
            raise ValueError(f"Directory not found: {faces_dir}")
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(faces_dir) 
                      if os.path.isdir(os.path.join(faces_dir, d))]
        
        if len(person_dirs) == 0:
            raise ValueError("No person directories found in faces directory")
        
        print(f"📊 Found {len(person_dirs)} persons")
        
        # Prepare training data
        training_data = []
        labels = []
        
        # Create label mapping
        for label_id, person_name in enumerate(person_dirs):
            person_path = os.path.join(faces_dir, person_name)
            
            # Get all images for this person
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"   Processing {person_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize to 200x200 (as per user's code)
                    img = cv2.resize(img, (200, 200))
                    
                    # Apply selected preprocessing method
                    img = self._apply_preprocessing(img)
                    
                    training_data.append(img)
                    labels.append(label_id)
            
            # Store label mapping (PersonName_RollNo format)
            self.labels_to_name[label_id] = person_name
        
        if len(training_data) == 0:
            raise ValueError("No training images found")
        
        # Convert to numpy arrays
        training_data = np.array(training_data, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"📸 Training with {len(training_data)} images from {len(self.labels_to_name)} persons")
        
        # Train the model
        self.model.train(training_data, labels)
        self.is_trained = True
        
        print(f"✅ Training completed successfully")
    
    def predict(self, face_image):
        """
        Predict the identity of a face.
        
        Args:
            face_image: Grayscale face image (numpy array)
        
        Returns:
            (person_name, confidence) tuple
            - person_name: Name of recognized person or "Unknown"
            - confidence: Confidence percentage (0-100)
        """
        if not self.is_trained:
            return "Unknown", 0
        
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 200x200 (as per user's code)
        face_image = cv2.resize(face_image, (200, 200))
        
        # Apply same preprocessing as training
        face_image = self._apply_preprocessing(face_image)
        
        # Predict
        label_id, distance = self.model.predict(face_image)
        
        # In LBPH: distance <= 110.0 is a strong match for the person
        if distance <= 110.0:
            confidence = int(max(45, min(99, 100 - (distance * 0.55))))
            person_name = self.labels_to_name.get(label_id, "Unknown")
            return person_name, confidence
        
        return "Unknown", 0
    
    def predict_with_distance(self, face_image):
        """
        Predict with raw distance value.
        
        Args:
            face_image: Face image (BGR or grayscale)
        
        Returns:
            (person_name, distance) tuple
        """
        if not self.is_trained:
            return "Unknown", 1000.0
        
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 200x200 (as per user's code)
        face_image = cv2.resize(face_image, (200, 200))
        
        # Apply same preprocessing as training
        face_image = self._apply_preprocessing(face_image)
        
        # Predict
        label_id, distance = self.model.predict(face_image)
        
        # Check threshold
        if distance <= 110.0:
            person_name = self.labels_to_name.get(label_id, "Unknown")
            return person_name, float(distance)
        
        return "Unknown", float(distance)
    
    def _apply_preprocessing(self, gray_img):
        """
        Apply the configured preprocessing method to an image.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Preprocessed image
        """
        if self.preprocessing_method == 'advanced':
            return preprocess_image_advanced(gray_img)
        elif self.preprocessing_method == 'simple':
            return preprocess_image_simple_clahe(gray_img)
        elif self.preprocessing_method == 'standard':
            return preprocess_image(gray_img, use_clahe=False)
        else:  # 'clahe' (default)
            return preprocess_image(gray_img, use_clahe=True)
    
    def save(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Save the LBPH model
        xml_path = model_path + ".xml"
        self.model.write(xml_path)
        
        # Save label mapping AND preprocessing method
        pkl_path = model_path + "_labels.pkl"
        model_data = {
            'labels_to_name': self.labels_to_name,
            'preprocessing_method': self.preprocessing_method  # Save preprocessing method
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {xml_path}")
        print(f"   Preprocessing method: {self.preprocessing_method}")
    
    def load(self, model_path):
        """
        Load a trained model.
        Automatically detects and loads the preprocessing method used during training.
        
        Args:
            model_path: Path to the model (without extension)
        """
        xml_path = model_path + ".xml"
        pkl_path = model_path + "_labels.pkl"
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model file not found: {xml_path}")
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Labels file not found: {pkl_path}")
        
        # Load the LBPH model
        self.model.read(xml_path)
        
        # Load label mapping and preprocessing method
        with open(pkl_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle both old format (dict only) and new format (with preprocessing_method)
        if isinstance(model_data, dict) and 'labels_to_name' in model_data:
            # New format: includes preprocessing method
            self.labels_to_name = model_data['labels_to_name']
            self.preprocessing_method = model_data.get('preprocessing_method', 'standard')
            print(f"✅ Model loaded from: {xml_path}")
            print(f"   Trained on {len(self.labels_to_name)} persons")
            print(f"   Preprocessing method: {self.preprocessing_method}")
        else:
            # Old format: only labels (backward compatibility)
            self.labels_to_name = model_data
            self.preprocessing_method = 'standard'  # Old models used standard preprocessing
            print(f"✅ Model loaded from: {xml_path} (Legacy format)")
            print(f"   Trained on {len(self.labels_to_name)} persons")
            print(f"   ⚠️  Using 'standard' preprocessing (old model format)")
        
        self.is_trained = True


# For backward compatibility
LBPModel = BasicLBPHRecognizer
