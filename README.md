# Digital Hajir: Face Recognition Attendance System

A modern attendance system using JavaScript face detection and Python face recognition.

## Features

- **Browser-based Face Detection**: Uses Pico.js, a lightweight custom face detection implementation in JavaScript
- **MySQL Database**: For storing user information and attendance records
- **Flask Backend**: Handles routing, user management, and data processing
- **Server-side Face Recognition**: Uses KNN algorithm for accurate face recognition
- **Attendance Tracking**: Automatically records attendance in CSV files
- **User Management**: Add, edit, and delete users with their face data
- **Responsive UI**: Works on desktop and mobile devices

## How to Use

### Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

### Taking Attendance

1. Click on "Take Attendance" on the home page
2. Click "TOGGLE WEBCAM" to start your camera
3. Click "MARK ATTENDANCE" to switch to attendance mode
4. The system will detect faces and mark attendance automatically
5. Recognized faces will be shown in the status panel
6. Click "View Full Attendance" to see the attendance list

### Adding New Users

1. Log in as an administrator
2. Click on "Add New User" on the home page
3. Fill in the user details and submit
4. Take 20 photos of the user's face for training

## Modes

- **Detection Mode**: Only detects faces without recognition (yellow boxes)
- **Attendance Mode**: Detects faces and marks attendance (green boxes)

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Backend**: Flask, Python
- **Face Detection**: Pico.js cascade-based detection (JavaScript)
- **Face Recognition**: K-Nearest Neighbors algorithm (Python)
- **Database**: MySQL

## License

This project is licensed under the MIT License. 