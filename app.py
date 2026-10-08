import cv2
import os
import shutil
import base64
import io
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from datetime import date, datetime
import numpy as np
import pandas as pd
import mysql.connector
import bcrypt
import re  # For email validation
# Emotion recognition (FER - CNN based) - optional, requires TensorFlow
try:
    from fer import FER
    _FER_AVAILABLE = True
except Exception:
    _FER_AVAILABLE = False
    print("⚠️ FER (emotion recognition) not available - emotion detection disabled.")

# Import Recognition Engines: Deep SFace (99.4% Deep Learning) with Basic LBPH fallback
from utils.recognizer_basic_lbph import BasicLBPHRecognizer as LBPModel
try:
    from utils.recognizer_deep_sface import DeepSFaceRecognizer
    _DEEP_SFACE_AVAILABLE = True
    print("🚀 Deep Learning SFace Engine available (99.4% ArcFace Accuracy)")
except Exception as e:
    _DEEP_SFACE_AVAILABLE = False
    print("⚠️ Deep SFace Engine not available, using LBPH fallback:", e)

import time

# Flask app setup
app = Flask(__name__)
app.secret_key = "3fc18c5457e07839473ddd0a81dfff24"

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for authentication
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(id):
    return User(id)

# MySQL database configuration
db_config = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "digitalhajir")
}

# Ensure attendance table has 'expression' column (auto-migration)
def ensure_expression_column():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW COLUMNS FROM attendance LIKE 'expression'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE attendance ADD COLUMN expression VARCHAR(32) NULL")
            conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Warning: could not ensure expression column exists: {e}")
# Function to insert user login data into the database
def insert_user(username, password):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Securely hash the password using bcrypt
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        sql = "INSERT INTO admin (username, password) VALUES (%s, %s)"
        values = (username, hashed_password)
        cursor.execute(sql, values)

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error inserting user:", e)


# Function to check user login credentials
def check_user_credentials(username, password):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        sql = "SELECT password FROM admin WHERE username = %s"
        values = (username,)
        cursor.execute(sql, values)

        user_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if user_data is not None:
            stored_hash = (user_data[0] or '').encode('utf-8')
            if stored_hash and bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return True
        return False
    except Exception as e:
        print("Error checking user credentials:", e)
        return False

    

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Ensure required directories and today's attendance file exist
if not os.path.isdir("Attendance"):
    os.makedirs("Attendance")
if not os.path.isdir("static/faces"):
    os.makedirs("static/faces")
if f"Attendance-{datetoday}.csv" not in os.listdir("Attendance"):
    with open(f"Attendance/Attendance-{datetoday}.csv", "w") as f:
        f.write("Name,Roll,Time")

# Initialize Haar cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    cascade_path = "static/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
BLUR_THRESHOLD = 80.0

def safe_imshow(window_name: str, frame) -> int:
    """Safely show frame in OpenCV window without crashing on macOS non-main threads."""
    try:
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1)
    except Exception:
        return -1

def safe_destroy_windows():
    """Safely close OpenCV GUI windows."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# Initialize FER emotion detector (once, only if TensorFlow is available)
emotion_detector = FER(mtcnn=False) if _FER_AVAILABLE else None

def _map_emotion_to_basic(label: str) -> str:
    try:
        lbl = (label or "").lower()
        if lbl == "happy":
            return "happy"
        if lbl in ("sad",):
            return "sad"
        return "neutral"
    except Exception:
        return "neutral"

# Get a number of total registered users
# def totalreg():
#     return len(os.listdir("static/faces"))
        
# Function to retrieve the total number of users from the database
def totalreg():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Execute SQL query to count the total number of users
        cursor.execute("SELECT COUNT(*) FROM users")

        # Fetch the result
        total_users = cursor.fetchone()[0]

        # Close the cursor and database connection
        cursor.close()
        conn.close()

        return total_users

    except Exception as e:
        print("Error fetching total number of users:", e)
        return None

# In your existing code, call the totalreg() function to get the total number of users
totalreg_count = totalreg()

if totalreg_count is not None:
    print("Total number of users:", totalreg_count)
else:
    print("Failed to retrieve total number of users")

# Extract the face from an image
def extract_faces(img):
    """Extract faces with tuned parameters suitable for varied lighting."""
    if img is None or not hasattr(img, 'size') or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Boost contrast for low-light scenarios

    face_points = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,   # Slightly faster scaling while keeping accuracy
        minNeighbors=5,    # More tolerant to lighting differences
        minSize=(28, 28),  # Allow slightly smaller detections
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Apply Non-Maximum Suppression to remove overlapping detections
    if len(face_points) > 1:
        face_points = apply_nms(face_points, overlap_threshold=0.3)
    
    return face_points


def apply_nms(boxes, overlap_threshold=0.3):
    """Apply Non-Maximum Suppression to remove overlapping face detections."""
    if len(boxes) == 0:
        return np.array([])
    
    # Convert to x1, y1, x2, y2 format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    
    # Sort by area (larger faces have priority)
    order = areas.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep only boxes with IoU less than threshold
        inds = np.where(iou <= overlap_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep]


def validate_email(email):
    """
    Validate email address format.
    
    Args:
        email: Email address string to validate
        
    Returns:
        True if valid email format, False otherwise
    """
    # Comprehensive email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not email or not isinstance(email, str):
        return False
    
    # Check basic format
    if not re.match(email_pattern, email):
        return False
    
    # Additional checks
    if email.count('@') != 1:
        return False
    
    # Split email into local and domain parts
    local, domain = email.split('@')
    
    # Check length constraints
    if len(local) == 0 or len(local) > 64:
        return False
    if len(domain) == 0 or len(domain) > 255:
        return False
    
    # Check for consecutive dots
    if '..' in email:
        return False
    
    # Check domain has at least one dot
    if '.' not in domain:
        return False
    
    return True


# Recognition backend (Deep SFace with LBPH fallback)
lbp_model = None

def ensure_lbp_model():
    """Load or train the face recognition model (Deep SFace if available, else LBPH)."""
    global lbp_model
    if lbp_model is not None:
        return lbp_model

    # Check if Deep SFace models exist
    if _DEEP_SFACE_AVAILABLE and os.path.exists("models/face_recognition_sface_2021dec.onnx"):
        try:
            print("🚀 Initializing Deep Learning SFace Recognizer...")
            sface = DeepSFaceRecognizer()
            if os.path.exists("static/sface_embeddings.pkl"):
                sface.load("static/sface_embeddings")
            elif os.path.exists("static/faces") and len(os.listdir("static/faces")) > 0:
                sface.train_from_directory("static/faces")
                sface.save("static/sface_embeddings")
            lbp_model = sface
            return lbp_model
        except Exception as e:
            print(f"⚠️ SFace initialization error: {e}, falling back to LBPH")

    model_path = os.path.join("static", "lbph_model")
    
    # Try to load existing model
    if os.path.exists(model_path + ".xml"):
        try:
            print(f"📂 Loading existing Basic LBPH model from: {model_path}.xml")
            model = LBPModel()
            model.load(model_path)
            lbp_model = model
            return lbp_model
        except Exception as e:
            print(f"⚠️ Error loading LBPH model: {e}")
            print("🔄 Will retrain...")
    
    # Train new model
    faces_root = os.path.join("static", "faces")
    if not os.path.exists(faces_root) or len(os.listdir(faces_root)) == 0:
        print(f"⚠️ No training data found in {faces_root}")
        return None
    
    print(f"🎓 Training new Basic LBPH model...")
    model = LBPModel()
    model.train_from_directory(faces_root)
    model.save(model_path)
    lbp_model = model
    return lbp_model

# Identify face using Basic LBPH model
def identify_face(facearray):
    """
    Identify a face using Basic LBPH recognizer.
    Returns formatted name or "Unknown".
    """
    # Reshape if needed
    if len(facearray.shape) == 1:
        side_length = int(np.sqrt(facearray.shape[0] // 3))
        facearray = facearray.reshape(side_length, side_length, 3)
    
    # Get model
    model = ensure_lbp_model()
    if model is None:
        return "Unknown"
    
    # Predict (model handles unknown detection internally)
    # Returns: (person_name, confidence) where confidence is 0-100
    start_time = time.time()
    person_name, confidence = model.predict(facearray)
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Check if unknown
    if person_name == "Unknown":
        print(f"❌ UNKNOWN person (confidence: {confidence}%)")
        return "Unknown"
    
    # Known person - log recognition
    print(f"✅ Recognized: {person_name} in {elapsed_ms:.1f}ms (confidence: {confidence}%)")
    
    # Format output: PersonName_RollNo -> PersonName (RollNo)
    try:
        name, roll = person_name.split('_')
        return f"{name} ({roll})"
    except:
        return person_name  # Return as is if format is different

    # (legacy training removed)

# Extract info from today's attendance from database
def extract_attendance():
    """Get today's attendance from database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Get today's attendance with student names
        query = """
            SELECT u.name, u.rollno, a.time 
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            WHERE a.date = CURDATE()
            ORDER BY a.time
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if results:
            names = [row[0] for row in results]
            rolls = [row[1] for row in results]
            times = [str(row[2]) for row in results]
            l = len(results)
            return names, rolls, times, l
        else:
            return [], [], [], 0
            
    except Exception as e:
        print(f"Error extracting attendance: {e}")
        return [], [], [], 0

# Function to add attendance of a user to database
def add_attendance(name, expression=None):
    """
    Add attendance for a user to database.
    Accepts:
      - "name_roll" (e.g., "John_12345")
      - "name (roll)" (e.g., "John (12345)")
      - "name" only (looks up roll from DB)
    """
    try:
        raw = str(name).strip()
        username = None
        userrollno = None

        # Case A: "name_roll"
        if '_' in raw and '(' not in raw and ')' not in raw:
            parts = raw.rsplit('_', 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                username = parts[0].strip()
                userrollno = parts[1].strip()

        # Case B: "name (roll)"
        if (username is None or userrollno is None) and '(' in raw and ')' in raw:
            try:
                left = raw[:raw.rfind('(')].strip()
                inside = raw[raw.rfind('(') + 1: raw.rfind(')')].strip()
                if inside.isdigit():
                    username = left
                    userrollno = inside
            except Exception:
                pass

        # Case C: only name -> fetch from DB
        if username is None or userrollno is None:
            user_details = get_user_by_name(raw)
            if user_details:
                username = str(user_details[1]).strip()
                userrollno = str(user_details[2]).strip()
            else:
                raise ValueError(f"User {raw} not found in database")

        if not username or not userrollno:
            raise ValueError(f"Invalid user data: name={username}, roll={userrollno}")

        # Normalize expression default
        if expression is None or str(expression).strip() == "":
            expression = "neutral"

        # Get student_id from database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM users WHERE rollno = %s", (userrollno,))
        result = cursor.fetchone()
        
        if not result:
            print(f"Error: Student with roll number {userrollno} not found in database")
            cursor.close()
            conn.close()
            return
        
        student_id = result[0]
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = date.today()
        
        # Check if attendance already marked today
        cursor.execute(
            "SELECT id FROM attendance WHERE student_id = %s AND date = %s",
            (student_id, current_date)
        )
        
        if cursor.fetchone():
            print(f"Attendance already marked for {username} (Roll: {userrollno})")
            cursor.close()
            conn.close()
            return
        
        # Insert attendance record (with expression if column exists)
        try:
            cursor.execute(
                """INSERT INTO attendance (student_id, date, time, marked_by, status, expression) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (student_id, current_date, current_time, 'system', 'present', expression)
            )
        except mysql.connector.Error:
            # Fallback if column missing
            cursor.execute(
                """INSERT INTO attendance (student_id, date, time, marked_by, status) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (student_id, current_date, current_time, 'system', 'present')
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Marked attendance for {username} (Roll: {userrollno}) | Expression: {expression}")

    except Exception as e:
        print(f"❌ Error marking attendance: {str(e)}")


# ==================== REPORTING & ANALYTICS FUNCTIONS ====================

# Get attendance for a specific date from database
def get_attendance_by_date(date_str):
    """Get attendance records for a specific date (format: MM_DD_YY or YYYY-MM-DD)"""
    try:
        # Convert date string to proper format
        if '_' in date_str:  # Format: MM_DD_YY
            date_obj = datetime.strptime(date_str, "%m_%d_%y")
            query_date = date_obj.strftime("%Y-%m-%d")
        else:  # Already in YYYY-MM-DD format
            query_date = date_str
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
            SELECT u.name as Name, u.rollno as Roll, a.time as Time, a.expression as Expression
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            WHERE a.date = %s
            ORDER BY a.time
        """
        cursor.execute(query, (query_date,))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if results:
            # Convert to DataFrame for compatibility (now includes Expression)
            df = pd.DataFrame(results, columns=['Name', 'Roll', 'Time', 'Expression'])
            return df
        return None
    except Exception as e:
        print(f"Error reading attendance for {date_str}: {e}")
        return None

# Get all attendance dates from database
def get_all_attendance_dates():
    """Get list of all dates with attendance records from database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Get distinct dates from attendance table
        cursor.execute("""
            SELECT DISTINCT date 
            FROM attendance 
            ORDER BY date DESC
        """)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        dates = []
        for row in results:
            date_obj = row[0]
            dates.append({
                'file_date': date_obj.strftime("%m_%d_%y"),  # For compatibility
                'display_date': date_obj.strftime("%d-%B-%Y"),
                'date_obj': date_obj
            })
        
        return dates
    except Exception as e:
        print(f"Error getting attendance dates: {e}")
        return []

# Calculate attendance percentage for a student from database
def calculate_attendance_percentage(rollno):
    """Calculate attendance percentage for a specific student"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Get total attendance days
        cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
        total_days = cursor.fetchone()[0]
        
        if total_days == 0:
            cursor.close()
            conn.close()
            return 0
        
        # Get student's present days
        cursor.execute("""
            SELECT COUNT(*) 
            FROM attendance a
            JOIN users u ON a.student_id = u.id
            WHERE u.rollno = %s AND a.status = 'present'
        """, (str(rollno),))
        present_days = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        percentage = (present_days / total_days) * 100
        return round(percentage, 2)
    except Exception as e:
        print(f"Error calculating attendance: {e}")
        return 0

# Get attendance statistics from database
def get_attendance_statistics():
    """Get overall attendance statistics from database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        total_registered = totalreg()
        
        # Today's attendance from database
        cursor.execute("""
            SELECT COUNT(*) 
            FROM attendance 
            WHERE date = CURDATE() AND status = 'present'
        """)
        today_present = cursor.fetchone()[0]
        
        today_absent = total_registered - today_present
        today_percentage = (today_present / total_registered * 100) if total_registered > 0 else 0
        
        # Get all students with their attendance percentage
        all_users = get_user_details()
        student_stats = []
        
        for user in all_users:
            user_id, name, rollno, email = user[0], user[1], user[2], user[3]
            percentage = calculate_attendance_percentage(str(rollno))
            student_stats.append({
                'id': user_id,
                'name': name,
                'rollno': rollno,
                'email': email,
                'percentage': percentage,
                'status': 'Good' if percentage >= 75 else 'Average' if percentage >= 60 else 'Poor'
            })
        
        cursor.close()
        conn.close()
        
        # Sort by percentage (lowest first for defaulters)
        student_stats.sort(key=lambda x: x['percentage'])
        
        # Get defaulters (below 75%)
        defaulters = [s for s in student_stats if s['percentage'] < 75]
        
        return {
            'total_registered': total_registered,
            'today_present': today_present,
            'today_absent': today_absent,
            'today_percentage': round(today_percentage, 2),
            'student_stats': student_stats,
            'defaulters': defaulters
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None

# Get attendance for date range
def get_attendance_date_range(start_date, end_date):
    """Get attendance records for a date range"""
    try:
        all_records = []
        files = os.listdir("Attendance")
        
        for file in files:
            if file.startswith("Attendance-") and file.endswith(".csv"):
                date_str = file.replace("Attendance-", "").replace(".csv", "")
                try:
                    file_date = datetime.strptime(date_str, "%m_%d_%y")
                    
                    # Check if date is in range
                    if start_date <= file_date <= end_date:
                        df = pd.read_csv(f"Attendance/{file}")
                        df['Date'] = file_date.strftime("%d-%B-%Y")
                        all_records.append(df)
                except:
                    pass
        
        if all_records:
            combined_df = pd.concat(all_records, ignore_index=True)
            return combined_df
        return None
    except Exception as e:
        print(f"Error getting date range attendance: {e}")
        return None

# ==================== END REPORTING & ANALYTICS ====================


# ==================== MANUAL ATTENDANCE MANAGEMENT ====================
# Note: Manual attendance management removed as per user request
# ==================== END MANUAL ATTENDANCE ====================


# ==================== SYSTEM MANAGEMENT FUNCTIONS ====================

# Change admin password
def change_admin_password(username, old_password, new_password):
    """Change admin password"""
    try:
        # First verify old password
        if not check_user_credentials(username, old_password):
            return False, "Current password is incorrect"
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Hash new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Update password
        cursor.execute("UPDATE admin SET password = %s WHERE username = %s", (hashed_password, username))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True, "Password changed successfully"
    except Exception as e:
        return False, f"Error changing password: {str(e)}"

# Retrain face recognition model
def retrain_face_model():
    """Retrain the Basic LBPH face recognition model"""
    try:
        faces_root = os.path.join("static", "faces")
        if not os.path.isdir(faces_root):
            return False, "Faces directory not found"
        
        # Create new Basic LBPH model
        model = LBPModel()
        model.train_from_directory(faces_root)
        
        # Save model
        model_path = os.path.join("static", "lbph_model")
        model.save(model_path)
        
        # Update global model
        global lbp_model
        lbp_model = model
        
        return True, "Model retrained successfully"
    except Exception as e:
        return False, f"Error retraining model: {str(e)}"

# Get system statistics
def get_system_stats():
    """Get system statistics"""
    try:
        stats = {
            'total_users': totalreg(),
            'total_attendance_days': len(get_all_attendance_dates()),
            'lbp_model_exists': os.path.exists("static/lbp_model_optimized.pkl") or os.path.exists("static/lbp_model.pkl"),
            'faces_folder_size': sum(len(files) for _, _, files in os.walk("static/faces")),
            'attendance_folder_size': len([f for f in os.listdir("Attendance") if f.endswith('.csv')])
        }
        return stats
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return None

# ==================== END SYSTEM MANAGEMENT ====================


# ==================== STUDENT AUTHENTICATION ====================

# Check student login credentials
def check_student_credentials(rollno, password):
    """Check if student credentials are valid"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Get student by roll number
        cursor.execute("SELECT id, name, rollno, email, password FROM users WHERE rollno = %s", (rollno,))
        user_data = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if user_data is None:
            return False, None
        
        # Check if password column exists and has value
        if len(user_data) >= 5 and user_data[4]:
            stored_hash = user_data[4].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                return True, {'id': user_data[0], 'name': user_data[1], 'rollno': user_data[2], 'email': user_data[3]}
        
        # If no password set, use default: rollno as password
        if password == str(rollno):
            return True, {'id': user_data[0], 'name': user_data[1], 'rollno': user_data[2], 'email': user_data[3]}
        
        return False, None
    except Exception as e:
        print(f"Error checking student credentials: {e}")
        return False, None

# Set student password
def set_student_password(rollno, new_password):
    """Set or update student password"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if password column exists
        cursor.execute("SHOW COLUMNS FROM users LIKE 'password'")
        if not cursor.fetchone():
            # Add password column if it doesn't exist
            cursor.execute("ALTER TABLE users ADD COLUMN password VARCHAR(255)")
        
        # Hash the password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Update password
        cursor.execute("UPDATE users SET password = %s WHERE rollno = %s", (hashed_password, rollno))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True, "Password set successfully"
    except Exception as e:
        return False, f"Error setting password: {str(e)}"

# ==================== END STUDENT AUTHENTICATION ====================


# Function to insert user details into the database
def insert_user_details(name, rollno, email):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Check if the roll number and email already exists in the database
        

        cursor.execute("SELECT * FROM users WHERE rollno = %s OR email = %s", (rollno, email))
        existing_user = cursor.fetchone()

        if existing_user:
            # If a user with the same roll number or email exists, display an error message
            if existing_user[1] == rollno:
                flash("Error: This roll number is already registered.", "danger")
            else:
                flash("Error: This email address is already taken.", "danger")
            return False

        # Insert user details into the database
        sql = "INSERT INTO users (name, rollno, email) VALUES (%s, %s, %s)"
        values = (name, rollno, email)
        cursor.execute(sql, values)
        conn.commit()

        cursor.close()
        conn.close()
        return True  # Return True on successful insertion

    except Exception as e:
        print("Error inserting user details:", e)
        return False

# Function to fetch user details from the database
def get_user_details(id=None):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        if id is not None:
            # Retrieve user details based on id
            sql = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql, (id,))
        else:
            # Retrieve all user details if id is not provided
            sql = "SELECT * FROM users"
            cursor.execute(sql)

        user_details = cursor.fetchall()
        cursor.close()
        conn.close()

        return user_details
    except Exception as e:
        print("Error fetching user details:", e)
        return []
# To retrieve all user details
all_users = get_user_details()

# To retrieve user details for a specific user with id = 1
user_details = get_user_details(id=1)



# Update user details in the database
def update_user_details(id, new_name, new_rollno, new_email):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # SQL statement to update user details using 'id' as the unique identifier
        sql = "UPDATE users SET name = %s, rollno = %s, email = %s WHERE id = %s"
        values = (new_name, new_rollno, new_email, id)
        cursor.execute(sql, values)
        conn.commit()

        cursor.close()
        conn.close()
        return True  # Return a success flag or message

    except Exception as e:
        print("Error updating user details:", e)

    return False  # Return an error flag or message

# Function to retrieve user details by ID
def get_user_by_id(id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Retrieve user details based on id
        sql = "SELECT * FROM users WHERE id = %s"
        cursor.execute(sql, (id,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        return user
    except Exception as e:
        print("Error fetching user details by ID:", e)
        return None

# Function to retrieve user details by name
def get_user_by_name(name):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Retrieve user details based on name
        sql = "SELECT * FROM users WHERE name = %s"
        cursor.execute(sql, (name,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        return user
    except Exception as e:
        print("Error fetching user details by name:", e)
        return None

# Function to delete a user by their ID
def delete_user_by_id(id):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # SQL statement to delete the user from the 'users' table
        sql = "DELETE FROM users WHERE id = %s"
        cursor.execute(sql, (id,))
        conn.commit()

        cursor.close()
        conn.close()

        flash("User deleted successfully", "success")
    except Exception as e:
        flash("Failed to delete user", "danger")


# ROUTING FUNCTIONS

##rote function to index
@app.route("/")
def index():
    return render_template("index.html")


###route function to login
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ""  # Initialize the message variable

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check user credentials (you may use check_user_credentials here)
        if check_user_credentials(username, password):
            user = User(username)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            message = 'Invalid username or password'

    return render_template('login.html', message=message)


##route function to logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


## route function for home
@app.route("/home")
@login_required
def home():
    names, rolls, times, l = extract_attendance()
    return render_template("home.html", names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

##route to use
@app.route("/use")
def use():
    return render_template("use.html")
##no train
@app.route("/notrain")
def notrain():
    return render_template("notrain.html")


## Route for in-browser live attendance / face detection
@app.route("/detect")
def detect():
    try:
        ensure_expression_column()
        ensure_lbp_model()
    except Exception as e:
        print("Model check:", e)
    return render_template("detect.html")


## Route when public user clicks start attendance -> in-browser detection
@app.route("/startuser", methods=["GET", "POST"])
def startuser():
    return redirect(url_for("detect"))


## Route when admin clicks take attendance -> in-browser detection
@app.route("/startadmin", methods=["GET", "POST"])
@login_required
def startadmin():
    return redirect(url_for("detect", ref="admin"))


## API endpoint called by in-browser detect.html to recognize face and mark attendance
@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "is_known": False, "label": "Unknown", "message": "No image provided"}), 400

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "is_known": False, "label": "Unknown", "message": "Invalid image"}), 400

        # Model recognition
        model = ensure_lbp_model()
        if model is None:
            return jsonify({"success": False, "is_known": False, "label": "Unknown", "confidence": 0, "message": "Model not ready"})

        person_name, confidence = model.predict(img)

        # Emotion recognition
        current_expr = "neutral"
        if _FER_AVAILABLE and emotion_detector is not None:
            try:
                face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                top = emotion_detector.top_emotion(face_rgb)
                if top and isinstance(top, tuple):
                    current_expr = _map_emotion_to_basic(top[0])
            except Exception:
                pass

        if person_name and person_name != "Unknown":
            try:
                name, roll = person_name.split('_')
                display_label = f"{name} (Roll: {roll})"
                first_name = name
                roll_no = roll
            except Exception:
                display_label = person_name
                first_name = person_name
                roll_no = ""

            should_mark = data.get("mark_attendance", False)
            if should_mark:
                add_attendance(person_name, expression=current_expr)

            return jsonify({
                "success": True,
                "is_known": True,
                "label": display_label,
                "name": first_name,
                "roll": roll_no,
                "confidence": round(float(confidence), 1) if confidence else 90.0,
                "expression": current_expr,
                "attendance_marked": should_mark,
                "message": f"Recognized: {display_label}"
            })
        else:
            return jsonify({
                "success": False,
                "is_known": False,
                "label": "Unknown",
                "name": "Unknown",
                "confidence": round(float(confidence), 1) if confidence else 0,
                "expression": current_expr,
                "attendance_marked": False,
                "message": "Face not recognized"
            })
    except Exception as e:
        return jsonify({"success": False, "is_known": False, "label": "Unknown", "message": str(e)}), 500
# Diagnostics: evaluate recognition on stored images (LBP only)
@app.route("/diagnostics")
@login_required
def diagnostics():
    try:
        # Ensure LBP model is loaded
        model = ensure_lbp_model()

        rows = []  # (true_label, predicted_label, distance)
        faces_root = "static/faces"
        if not os.path.isdir(faces_root):
            flash("No faces directory found.", "warning")
            return redirect(url_for("home"))

        # For each user, test on up to 3 images
        for user_dir in os.listdir(faces_root):
            user_path = os.path.join(faces_root, user_dir)
            if not os.path.isdir(user_path):
                continue
            count = 0
            for imgname in os.listdir(user_path):
                if count >= 3:
                    break
                img_path = os.path.join(user_path, imgname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # mimic runtime preprocessing similar to runtime path
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    crop = img
                else:
                    (x, y, w, h) = faces[0]
                    pad = int(0.08 * max(w, h))
                    x0 = max(0, x - pad)
                    y0 = max(0, y - pad)
                    x1 = min(img.shape[1], x + w + pad)
                    y1 = min(img.shape[0], y + h + pad)
                    crop = img[y0:y1, x0:x1]

                label, distance = model.predict(crop, threshold=None)
                rows.append((user_dir, label or "None", round(float(distance), 3)))
                count += 1

        # Build a simple HTML table
        html = ["<h2>Diagnostics</h2>", "<table border='1' cellpadding='6' cellspacing='0'>",
                "<tr><th>True Label</th><th>Predicted</th><th>Distance</th></tr>"]
        for t, p, d in rows:
            color = "#d4edda" if p == t else "#f8d7da"
            html.append(f"<tr style='background:{color}'><td>{t}</td><td>{p}</td><td>{d}</td></tr>")
        html.append("</table>")
        html.append("<p>Lower distance generally indicates a closer match.</p>")
        return "".join(html)
    except Exception as e:
        return f"Diagnostics error: {str(e)}"


## In-browser Face Capture View
@app.route("/face_capture")
@login_required
def face_capture_view():
    username = request.args.get("username")
    rollno = request.args.get("rollno")
    if not username or not rollno:
        flash("Missing username or roll number.", "danger")
        return redirect(url_for("home"))
    return render_template("face_capture.html", username=username, rollno=rollno)


## API endpoint to receive captured face images from browser
@app.route("/capture_face", methods=["POST"])
def capture_face():
    try:
        data = request.get_json()
        if not data or "image" not in data or "username" not in data or "rollno" not in data:
            return jsonify({"success": False, "message": "Missing required data"}), 400

        username = data["username"]
        rollno = data["rollno"]
        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "message": "Invalid image"}), 400

        userimagefolder = os.path.join("static", "faces", f"{username}_{rollno}")
        if not os.path.exists(userimagefolder):
            os.makedirs(userimagefolder)

        existing = [f for f in os.listdir(userimagefolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(existing)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (200, 200))
        filename = f"{username}_{count}.jpg"
        cv2.imwrite(os.path.join(userimagefolder, filename), resized)

        new_count = count + 1

        if new_count >= 20:
            try:
                global lbp_model
                if _DEEP_SFACE_AVAILABLE and os.path.exists("models/face_recognition_sface_2021dec.onnx"):
                    sface = DeepSFaceRecognizer()
                    sface.train_from_directory("static/faces")
                    sface.save("static/sface_embeddings")
                    lbp_model = sface
                    print(f"🚀 Deep SFace retrained with {new_count} images for {username}")
                else:
                    model = LBPModel()
                    model.train_from_directory(os.path.join("static", "faces"))
                    model.save(os.path.join("static", "lbph_model"))
                    lbp_model = model
                    print(f"✅ LBPH model retrained with {new_count} images for {username}")
            except Exception as e:
                print("Retrain error:", e)

        return jsonify({"success": True, "count": new_count})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


### route function to add new user
@app.route("/add", methods=["GET", "POST"])
@login_required
def add():
    if request.method == "POST":
        newusername = request.form.get("newusername")
        newuserrollno = request.form.get("newuserrollno")
        useremail = request.form.get("newuseremail")

        if newusername and newuserrollno and useremail:
            # Validate email format
            if not validate_email(useremail):
                flash("❌ Invalid email address. Please enter a valid email (e.g., student@example.com)", "danger")
                return redirect(url_for("home"))
            
            # Insert user details into the database
            if insert_user_details(newusername, newuserrollno, useremail):
                userimagefolder = os.path.join("static", "faces", f"{newusername}_{newuserrollno}")
                if not os.path.isdir(userimagefolder):
                    os.makedirs(userimagefolder)

                # Open browser face capture
                return redirect(url_for("face_capture_view", username=newusername, rollno=newuserrollno))
            else:
                flash("❌ User details already exist or database insertion failed.", "danger")
                return redirect(url_for("home"))
        else:
            flash("❌ Please fill in all fields.", "danger")
            return redirect(url_for("home"))
    return redirect(url_for("home"))




# Route to display user details
@app.route("/userdetails")
@login_required
def userdetails():
    user_details = get_user_details()
    return render_template("userdetails.html", user_details=user_details)


## Route to add more images for existing user
@app.route("/add_more_images/<int:id>", methods=["GET", "POST"])
@login_required
def add_more_images(id):
    """Form to specify how many additional images to capture"""
    if request.method == "POST":
        num_images = request.form.get("num_images")
        if num_images and num_images.isdigit():
            num_images = int(num_images)
            if 1 <= num_images <= 100:
                return redirect(url_for("capture_more_images", id=id, count=num_images))
            else:
                flash("Please enter a number between 1 and 100", "danger")
        else:
            flash("Please enter a valid number", "danger")
    
    # Get user details for display
    user = get_user_details(id=id)
    if not user:
        flash("User not found", "danger")
        return redirect(url_for("userdetails"))
    
    user_info = {
        'id': user[0][0],
        'name': user[0][1],
        'rollno': user[0][2]
    }
    
    return render_template("add_more_images_form.html", user=user_info)


## Route to capture additional images for existing user
@app.route("/capture_more_images/<int:id>/<int:count>")
@login_required
def capture_more_images(id, count):
    """Capture additional training images for an existing user using in-browser webcam"""
    user = get_user_details(id=id)
    if not user:
        flash("User not found", "danger")
        return redirect(url_for("userdetails"))
    
    username = user[0][1]
    rollno = user[0][2]
    return redirect(url_for("face_capture_view", username=username, rollno=rollno))


## a route to edit users
# Route to edit user details
@app.route("/edit_user/<int:id>", methods=["GET", "POST"])
@login_required
def edit_user(id):
    if request.method == "POST":
        new_name = request.form["new_name"]
        new_rollno = request.form["new_rollno"]
        new_email = request.form["new_email"]
        
        # Validate email format
        if not validate_email(new_email):
            flash("❌ Invalid email address. Please enter a valid email (e.g., student@example.com)", "danger")
            return redirect(url_for("edit_user", id=id))
        
        # Call a function to update the user's details in the database
        if update_user_details(id, new_name, new_rollno, new_email):
            flash("User details updated successfully", "success")
            return redirect(url_for("userdetails"))
        else:
            flash("Failed to update user details", "danger")

    # Fetch a single user record (tuple) and convert it to a dict with the
    # keys that the template expects: id, user_name, user_rollno, user_email
    user_row = get_user_by_id(id)
    if user_row is None:
        flash("User not found", "danger")
        return redirect(url_for("userdetails"))

    # Assume the users table columns are in order: id, name, rollno, email
    try:
        user = {
            'id': user_row[0],
            'user_name': user_row[1],
            'user_rollno': user_row[2],
            'user_email': user_row[3]
        }
    except Exception:
        # Fallback: if tuple layout differs, pass minimal mapping so template errors are clearer
        user = {
            'id': id,
            'user_name': user_row[1] if len(user_row) > 1 else '',
            'user_rollno': user_row[2] if len(user_row) > 2 else '',
            'user_email': user_row[3] if len(user_row) > 3 else ''
        }

    return render_template("edit_user.html", user=user)

# Example route for updating user details
@app.route("/edit_user/<int:id>/update", methods=["POST"])
@login_required
def update_user(id):
    new_name = request.form["new_name"]
    new_rollno = request.form["new_rollno"]
    new_email = request.form["new_email"]

    if update_user_details(id, new_name, new_rollno, new_email):
        flash("User details updated successfully", "success")
    else:
        flash("Failed to update user details", "danger")

    return redirect(url_for("userdetails"))



# Route to delete a user (confirmation page)
@app.route("/delete_user_confirmation/<int:id>", methods=["GET"])
@login_required
def delete_user_confirmation(id):
    # Fetch the user details based on the provided ID and pass them to the confirmation template.
    user = get_user_by_id(id)  # Implement your function to retrieve user details by ID.
    return render_template("delete_user_confirmation.html", user=user)


# Route to delete a user (action)
@app.route("/delete_user/<int:id>", methods=["POST"])
@login_required
def delete_user(id):
    user = get_user_details(id)
    if user is None:
        flash("User not found", "danger")
    else:
        # Also remove the user's face images from the static/faces folder BEFORE deleting from DB
        try:
            user_name = user[0][1]  # name is the second column
            user_rollno = user[0][2]  # rollno is the third column
            # Folder format is: name_rollno (same as when created in /add route)
            user_faces_folder = f"{user_name}_{user_rollno}"
            user_faces_path = os.path.join("static", "faces", user_faces_folder)
            
            if os.path.exists(user_faces_path):
                # Remove all files in the folder
                shutil.rmtree(user_faces_path)
                print(f"✅ Deleted face folder: {user_faces_path}")
            else:
                print(f"⚠️ Face folder not found: {user_faces_path}")
        except Exception as e:
            print(f"❌ Error removing user face images: {e}")
        
        # Call the function to delete the user from the database
        delete_user_by_id(id)
        flash("User and face data deleted successfully", "success")
    return redirect(url_for("userdetails"))


# ==================== REPORTING & ANALYTICS ROUTES ====================

# Route for attendance reports
@app.route("/reports")
@login_required
def reports():
    """Main reports page with statistics"""
    stats = get_attendance_statistics()
    dates = get_all_attendance_dates()
    
    return render_template("reports.html", 
                         stats=stats, 
                         dates=dates,
                         datetoday2=datetoday2)

# Route to view attendance by date
@app.route("/view_attendance/<date_str>")
@login_required
def view_attendance_by_date(date_str):
    """View attendance for a specific date"""
    df = get_attendance_by_date(date_str)
    
    if df is not None:
        # Convert to readable date
        try:
            date_obj = datetime.strptime(date_str, "%m_%d_%y")
            display_date = date_obj.strftime("%d-%B-%Y")
        except:
            display_date = date_str
        
        attendance_data = df.to_dict('records')
        return render_template("view_attendance.html", 
                             attendance=attendance_data,
                             date=display_date,
                             date_str=date_str,
                             count=len(df))
    else:
        flash("No attendance records found for this date", "warning")
        return redirect(url_for("reports"))

# Route to export attendance to Excel
@app.route("/export_attendance/<format>")
@login_required
def export_attendance(format):
    """Export attendance to Excel or CSV"""
    try:
        os.makedirs("Attendance", exist_ok=True)

        if format == "excel":
            # Create Excel file with all attendance data
            dates = get_all_attendance_dates()

            output_filename = f"Attendance_Report_{datetoday}.xlsx"
            output_path = os.path.join("Attendance", output_filename)

            # Create a Pandas Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Write each date's attendance to a separate sheet
                for date_info in dates[:10]:  # Limit to last 10 dates
                    df = get_attendance_by_date(date_info['file_date'])
                    if df is not None:
                        sheet_name = date_info['display_date'][:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Add summary sheet with all students and their percentages
                all_users = get_user_details()
                summary_data = []
                for user in all_users:
                    user_id, name, rollno, email = user[0], user[1], user[2], user[3]
                    percentage = calculate_attendance_percentage(str(rollno))
                    summary_data.append({
                        'Name': name,
                        'Roll No': rollno,
                        'Email': email,
                        'Attendance %': percentage
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            flash("Attendance exported to the Attendance folder as "
                  f"{output_filename}", "success")
            return redirect(url_for("reports"))

        elif format == "csv":
            # Export today's attendance as CSV
            df = get_attendance_by_date(datetoday)
            if df is not None:
                output_filename = f"Attendance_Export_{datetoday}.csv"
                output_path = os.path.join("Attendance", output_filename)
                df.to_csv(output_path, index=False)
                flash("Today's attendance exported to the Attendance folder as "
                      f"{output_filename}", "success")
            else:
                flash("No attendance data for today", "warning")
            return redirect(url_for("reports"))
            
    except Exception as e:
        flash(f"Error exporting attendance: {str(e)}", "danger")
        return redirect(url_for("reports"))

# Route for defaulters list
@app.route("/defaulters")
@login_required
def defaulters():
    """Show list of students with low attendance"""
    stats = get_attendance_statistics()
    
    if stats:
        defaulters_list = stats['defaulters']
        return render_template("defaulters.html", 
                             defaulters=defaulters_list,
                             datetoday2=datetoday2)
    else:
        flash("Error loading defaulters data", "danger")
        return redirect(url_for("home"))

# ==================== END REPORTING ROUTES ====================


# ==================== MANUAL ATTENDANCE ROUTES ====================
# Note: Manual attendance routes removed as per user request
# ==================== END MANUAL ATTENDANCE ROUTES ====================


# ==================== SYSTEM MANAGEMENT ROUTES ====================

# Route for system settings page
@app.route("/settings")
@login_required
def settings():
    """System settings and management page"""
    stats = get_system_stats()
    return render_template("settings.html", 
                         stats=stats,
                         datetoday2=datetoday2)

# Route to change admin password
@app.route("/change_password", methods=["POST"])
@login_required
def change_password():
    """Change admin password"""
    try:
        username = session.get('_user_id', 'admin')  # Get current username from session
        old_password = request.form.get("old_password")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")
        
        if new_password != confirm_password:
            flash("New passwords do not match", "danger")
            return redirect(url_for("settings"))
        
        if len(new_password) < 6:
            flash("Password must be at least 6 characters long", "danger")
            return redirect(url_for("settings"))
        
        success, message = change_admin_password(username, old_password, new_password)
        
        if success:
            flash(message, "success")
        else:
            flash(message, "danger")
            
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
    
    return redirect(url_for("settings"))

# Route to retrain model
@app.route("/retrain_model", methods=["POST"])
@login_required
def retrain_model_route():
    """Retrain face recognition model"""
    try:
        success, message = retrain_face_model()
        
        if success:
            flash(message, "success")
        else:
            flash(message, "danger")
            
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
    
    return redirect(url_for("settings"))

# ==================== END SYSTEM MANAGEMENT ROUTES ====================


# ==================== STUDENT ROUTES ====================

# Student login page
@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    """Student login page"""
    if request.method == "POST":
        rollno = request.form.get("rollno")
        password = request.form.get("password")
        
        success, student_data = check_student_credentials(rollno, password)
        
        if success:
            # Store student info in session
            session['student_id'] = student_data['id']
            session['student_rollno'] = student_data['rollno']
            session['student_name'] = student_data['name']
            session['is_student'] = True
            
            flash(f"Welcome, {student_data['name']}!", "success")
            return redirect(url_for('student_dashboard'))
        else:
            flash("Invalid roll number or password", "danger")
    
    return render_template("student_login.html")

# Student dashboard
@app.route("/student/dashboard")
def student_dashboard():
    """Student dashboard - view own attendance"""
    # Check if student is logged in
    if not session.get('is_student'):
        flash("Please login first", "warning")
        return redirect(url_for('student_login'))
    
    student_rollno = session.get('student_rollno')
    student_name = session.get('student_name')
    
    # Get student's attendance data
    attendance_percentage = calculate_attendance_percentage(str(student_rollno))
    
    # Get attendance history
    attendance_dates = get_all_attendance_dates()
    attendance_history = []
    present_days = 0
    total_days = len(attendance_dates)
    
    for date_info in attendance_dates:
        df = get_attendance_by_date(date_info['file_date'])
        if df is not None:
            is_present = str(student_rollno) in df["Roll"].astype(str).values
            if is_present:
                present_days += 1
                row = df[df["Roll"].astype(str) == str(student_rollno)].iloc[0]
                time_marked = row["Time"] if "Time" in row else "-"
                expr_marked = row["Expression"] if "Expression" in row else None
                attendance_history.append({
                    'date': date_info['display_date'],
                    'status': 'Present',
                    'time': time_marked,
                    'expression': expr_marked or 'neutral'
                })
            else:
                attendance_history.append({
                    'date': date_info['display_date'],
                    'status': 'Absent',
                    'time': '-',
                    'expression': '-'
                })
    
    return render_template("student_dashboard.html",
                         student_name=student_name,
                         student_rollno=student_rollno,
                         attendance_percentage=attendance_percentage,
                         present_days=present_days,
                         total_days=total_days,
                         attendance_history=attendance_history,
                         datetoday2=datetoday2)

# Student logout
@app.route("/student/logout")
def student_logout():
    """Student logout"""
    session.pop('student_id', None)
    session.pop('student_rollno', None)
    session.pop('student_name', None)
    session.pop('is_student', None)
    flash("Logged out successfully", "info")
    return redirect(url_for('index'))

# Student change password
@app.route("/student/change_password", methods=["GET", "POST"])
def student_change_password():
    """Student change password page"""
    # Check if student is logged in
    if not session.get('is_student'):
        flash("Please login first", "warning")
        return redirect(url_for('student_login'))
    
    if request.method == "POST":
        student_rollno = session.get('student_rollno')
        old_password = request.form.get("old_password")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")
        
        # Validate passwords
        if new_password != confirm_password:
            flash("New passwords do not match", "danger")
            return redirect(url_for('student_change_password'))
        
        if len(new_password) < 6:
            flash("Password must be at least 6 characters long", "danger")
            return redirect(url_for('student_change_password'))
        
        # Verify old password
        success, student_data = check_student_credentials(student_rollno, old_password)
        
        if not success:
            flash("Current password is incorrect", "danger")
            return redirect(url_for('student_change_password'))
        
        # Set new password
        success, message = set_student_password(student_rollno, new_password)
        
        if success:
            flash("Password changed successfully! Please login again.", "success")
            # Logout student
            session.pop('student_id', None)
            session.pop('student_rollno', None)
            session.pop('student_name', None)
            session.pop('is_student', None)
            return redirect(url_for('student_login'))
        else:
            flash(message, "danger")
            return redirect(url_for('student_change_password'))
    
    return render_template("student_change_password.html",
                         student_name=session.get('student_name'),
                         student_rollno=session.get('student_rollno'),
                         datetoday2=datetoday2)

# ==================== END STUDENT ROUTES ====================

# ==================== MODEL RETRAINING ROUTE ====================

@app.route('/retrain_model')
@login_required
def retrain_model():
    """
    Retrain the Basic LBPH model.
    Call this after adding new users or updating face images.
    """
    global lbp_model
    
    try:
        # Delete existing model files
        model_xml = os.path.join("static", "lbph_model.xml")
        model_pkl = os.path.join("static", "lbph_model_labels.pkl")
        
        if os.path.exists(model_xml):
            os.remove(model_xml)
            print("🗑️ Deleted old model.xml")
        if os.path.exists(model_pkl):
            os.remove(model_pkl)
            print("🗑️ Deleted old labels.pkl")
        
        # Reset model
        lbp_model = None
        
        # Retrain by calling ensure_lbp_model
        print("🔄 Starting model retraining...")
        ensure_lbp_model()
        
        flash("Model retrained successfully! All users are now recognized.", "success")
        print("✅ Model retrained successfully")
        
    except Exception as e:
        flash(f"Error retraining model: {str(e)}", "danger")
        print(f"❌ Error retraining model: {e}")
    
    return redirect(url_for('home'))


# ==================== END MODEL RETRAINING ====================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
