"""
Deep Learning Face Recognizer (SFace + YuNet)
Using OpenCV's Deep Neural Network (DNN) module with CosFace/ArcFace Embeddings.
Provides 99.4% accuracy, invariant to lighting, glasses, and subtle facial angles.
"""

import os
import cv2
import pickle
import numpy as np


class DeepSFaceRecognizer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.sface_model_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
        self.yunet_model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        
        self.detector = None
        self.recognizer = None
        self.embeddings_db = {}  # {person_name: list of 128-D feature vectors}
        self.is_trained = False
        self.cosine_threshold = 0.363  # Standard OpenCV SFace cosine similarity threshold
        
        self._init_models()

    def _init_models(self):
        """Initialize YuNet detector and SFace recognizer."""
        if not os.path.exists(self.sface_model_path) or not os.path.exists(self.yunet_model_path):
            print(f"⚠️ SFace/YuNet model files not found in {self.models_dir}")
            return
        
        try:
            # Face detector: YuNet
            self.detector = cv2.FaceDetectorYN.create(
                self.yunet_model_path,
                "",
                (320, 320),
                0.6,   # Score threshold
                0.3,   # NMS threshold
                5000   # top_k
            )
            
            # Deep face feature extractor: SFace (128-D vector)
            self.recognizer = cv2.FaceRecognizerSF.create(self.sface_model_path, "")
            print("🚀 Deep Learning SFace Recognizer (99.4% Accuracy) loaded successfully!")
        except Exception as e:
            print(f"❌ Error initializing SFace models: {e}")

    def extract_feature(self, img_bgr):
        """
        Extract 128-dimensional deep feature vector from a face image.
        Uses YuNet landmark alignment when possible, or direct 112x112 resize fallback.
        """
        if self.recognizer is None or img_bgr is None or img_bgr.size == 0:
            return None

        # Convert grayscale to BGR if necessary
        if len(img_bgr.shape) == 2 or img_bgr.shape[2] == 1:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        h, w = img_bgr.shape[:2]
        
        # Try detecting face and facial landmarks for 5-point affine alignment
        if self.detector is not None and h >= 30 and w >= 30:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(img_bgr)
                if faces is not None and len(faces) > 0:
                    # Select largest face
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    aligned = self.recognizer.alignCrop(img_bgr, best_face)
                    return self.recognizer.feature(aligned)
            except Exception:
                pass
        
        # Fallback: direct crop resize to 112x112 for SFace
        try:
            resized = cv2.resize(img_bgr, (112, 112))
            return self.recognizer.feature(resized)
        except Exception:
            return None

    def train_from_directory(self, faces_dir="static/faces"):
        """
        Scan faces directory and compute deep embeddings for each registered person.
        """
        if not os.path.exists(faces_dir):
            raise ValueError(f"Directory not found: {faces_dir}")

        person_dirs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
        if len(person_dirs) == 0:
            raise ValueError("No person directories found in faces directory")

        print(f"🎓 Training Deep SFace on {len(person_dirs)} persons...")
        self.embeddings_db = {}

        for person_name in person_dirs:
            person_path = os.path.join(faces_dir, person_name)
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            features = []
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    feat = self.extract_feature(img)
                    if feat is not None:
                        features.append(feat)

            if features:
                self.embeddings_db[person_name] = features
                print(f"   👤 {person_name}: {len(features)} deep face vectors extracted")

        self.is_trained = len(self.embeddings_db) > 0
        print(f"✅ Deep SFace trained on {len(self.embeddings_db)} persons successfully!")
        return self.is_trained

    def predict(self, face_image):
        """
        Predict identity with deep cosine similarity.
        Returns: (person_name, confidence_percent)
        """
        if not self.is_trained or self.recognizer is None:
            return "Unknown", 0

        target_feature = self.extract_feature(face_image)
        if target_feature is None:
            return "Unknown", 0

        best_person = "Unknown"
        best_score = -1.0

        for person_name, feature_list in self.embeddings_db.items():
            for known_feat in feature_list:
                score = self.recognizer.match(known_feat, target_feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_person = person_name

        # Standard SFace match threshold is >= 0.363
        if best_score >= self.cosine_threshold:
            # Map score (0.363 to 0.85+) to confidence (60% to 99%)
            confidence = int(min(99, max(60, 60 + ((best_score - 0.363) / 0.45) * 39)))
            return best_person, confidence

        return "Unknown", 0

    def save(self, model_prefix="static/sface_embeddings"):
        """Save the deep embeddings database to disk."""
        pkl_path = model_prefix + ".pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self.embeddings_db, f)
        print(f"💾 Deep SFace embeddings saved to: {pkl_path}")

    def load(self, model_prefix="static/sface_embeddings"):
        """Load embeddings database from disk."""
        pkl_path = model_prefix + ".pkl"
        if not os.path.exists(pkl_path):
            # If embeddings pkl does not exist yet, train from directory
            if os.path.exists("static/faces"):
                self.train_from_directory("static/faces")
                self.save(model_prefix)
                return
            raise FileNotFoundError(f"Embeddings file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            self.embeddings_db = pickle.load(f)
        self.is_trained = len(self.embeddings_db) > 0
        print(f"✅ Deep SFace embeddings loaded ({len(self.embeddings_db)} persons registered)")
