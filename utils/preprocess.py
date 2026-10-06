"""
Face detection and preprocessing helper functions
Based on user's original working code
"""

import os
import cv2 as cv
import numpy as np

# Load HAAR face classifier
face_cascade_path = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(face_cascade_path):
    face_cascade_path = os.path.join(os.path.dirname(__file__), "..", "static", "haarcascade_frontalface_default.xml")
face_cascade = cv.CascadeClassifier(face_cascade_path)


def face_detector(gray_img):
    """
    Detect face in grayscale image.
    
    Arguments:
        gray_img: Image in Grayscale Format
    
    Returns:
        roi: Region of interest (Face) resized to 200x200
        coord: Coordinates (x, y, w, h)
    """
    if gray_img is None or not hasattr(gray_img, 'size') or gray_img.size == 0:
        return None, (None, None, None, None)

    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    if faces is None or len(faces) == 0:
        return None, (None, None, None, None)
    
    for (x, y, w, h) in faces:
        cv.rectangle(gray_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_face = gray_img[y:y+h, x:x+w]
        roi = cv.resize(cropped_face, (200, 200))
        coord = (x, y, w, h)
        break  # Only process first face

    return roi, coord


def preprocess_image(gray_img, use_clahe=True):
    """
    Enhanced preprocessing with CLAHE for better face recognition.
    
    - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Splits image into left and right halves for better local contrast
    - Applies bilateral filtering for noise reduction
    - Optional Gaussian blur for smoothing
    
    Arguments:
        gray_img: Grayscale face image
        use_clahe: Whether to use CLAHE (default True, better than standard equalization)
    
    Returns:
        Preprocessed image with enhanced contrast and reduced noise
    """
    height, width = gray_img.shape[:2]

    # Split image into left and right halves
    midX = int(width / 2)
    leftSide = gray_img[int(0):height, int(0):midX]
    rightSide = gray_img[int(0):height, midX:width]

    if use_clahe:
        # CLAHE: Better than standard histogram equalization
        # clipLimit: Threshold for contrast limiting (prevents over-amplification)
        # tileGridSize: Size of grid for histogram equalization (smaller = more local)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equL = clahe.apply(leftSide)
        equR = clahe.apply(rightSide)
    else:
        # Fallback to standard histogram equalization
        equL = cv.equalizeHist(leftSide)
        equR = cv.equalizeHist(rightSide)

    # Combine both halves
    img_equalized = np.concatenate((equL, equR), axis=1)
    
    # Apply bilateral filter to reduce noise while preserving edges
    # This is crucial for face recognition accuracy
    img = cv.bilateralFilter(img_equalized, 15, 75, 75)
    
    return img


def preprocess_image_advanced(gray_img):
    """
    Advanced preprocessing with multiple enhancement techniques.
    
    Includes:
    - CLAHE for adaptive contrast enhancement
    - Gaussian blur for noise reduction
    - Image normalization
    - Bilateral filtering
    
    Arguments:
        gray_img: Grayscale face image
    
    Returns:
        Preprocessed image with maximum enhancement
    """
    # Step 1: Slight Gaussian blur to reduce high-frequency noise
    img_blurred = cv.GaussianBlur(gray_img, (3, 3), 0)
    
    # Step 2: Apply CLAHE to full image
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_blurred)
    
    # Step 3: Split and apply additional CLAHE to halves for face symmetry
    height, width = img_clahe.shape[:2]
    midX = int(width / 2)
    leftSide = img_clahe[int(0):height, int(0):midX]
    rightSide = img_clahe[int(0):height, midX:width]
    
    clahe_local = cv.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    equL = clahe_local.apply(leftSide)
    equR = clahe_local.apply(rightSide)
    
    img_combined = np.concatenate((equL, equR), axis=1)
    
    # Step 4: Bilateral filter for edge-preserving smoothing
    img_filtered = cv.bilateralFilter(img_combined, 9, 50, 50)
    
    # Step 5: Normalize to [0, 255] range
    img_normalized = cv.normalize(img_filtered, None, 0, 255, cv.NORM_MINMAX)
    
    return img_normalized


def preprocess_image_simple_clahe(gray_img):
    """
    Simple CLAHE preprocessing without splitting.
    Faster and works well for well-lit conditions.
    
    Arguments:
        gray_img: Grayscale face image
    
    Returns:
        CLAHE-enhanced image
    """
    # Apply CLAHE to entire image
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_img)
    
    # Light bilateral filtering
    img_filtered = cv.bilateralFilter(img_clahe, 9, 50, 50)
    
    return img_filtered
