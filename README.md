# 🎓 Digital Hajir: AI Face Recognition Attendance System

A modern, full-stack Face Recognition Attendance System built with **Flask**, **OpenCV (LBPH)**, **Pico.js**, and **MySQL**, containerized with **Docker**.

---

## 🌟 Key Features

- 📷 **In-Browser Face Detection & Capture**: Uses client-side JavaScript (`pico.js`) and HTML5 `getUserMedia` — no server desktop window popups or Cocoa/GUI crashes.
- 🏷️ **Real-Time Dynamic Head Badges**: Live recognition overlay displaying `🟢 Name (Match %)` for recognized students and `❓ Unknown Person` for unregistered faces.
- ⚡ **Lightweight & Fast Recognition**: Powered by OpenCV's Local Binary Patterns Histograms (LBPH) — runs smoothly on any laptop without GPU requirements.
- 🗄️ **Visual Database Management (phpMyAdmin)**: Inspect, query, and edit MySQL database tables directly via a web browser.
- 📊 **Automated Attendance Logging**: Logs timestamp, date, student details, and attendance state to MySQL and exportable records.
- 🐳 **Dockerized**: 1-click multi-container setup (Web App + MySQL 8.0 + phpMyAdmin).

---

## 🔑 Default Credentials

| Service | URL | Credentials |
| :--- | :--- | :--- |
| **Web Application** | [http://localhost:5001](http://localhost:5001) | **Username:** `admin` <br> **Password:** `admin123` |
| **Database UI (phpMyAdmin)** | [http://localhost:8080](http://localhost:8080) | **Server:** `db` <br> **Username:** `root` <br> **Password:** *(leave blank)* |

---

## 🚀 How to Run the Project

### Method 1: Docker (Recommended — Easiest for Everyone)

> **Requirement:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Final-Year-Project-Python
   ```

2. **Start all services:**
   ```bash
   docker compose up --build -d
   ```

3. **Open in your browser:**
   - Web App: **[http://localhost:5001](http://localhost:5001)**
   - Database UI: **[http://localhost:8080](http://localhost:8080)**

To stop the containers:
```bash
docker compose down
```

---

### Method 2: Local Python Setup (Hybrid Development)

If you prefer running the Python app locally with live-reloading:

1. **Start MySQL Database & phpMyAdmin in Docker:**
   ```bash
   docker compose up db phpmyadmin -d
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Initialize Admin Account:**
   ```bash
   python utils/create_admin.py
   ```

5. **Start the Flask Web App:**
   ```bash
   PORT=5001 python app.py
   ```

6. **Open in browser:**
   - Web App: **[http://localhost:5001](http://localhost:5001)**
   - phpMyAdmin: **[http://localhost:8080](http://localhost:8080)**

---

## 📖 How to Use the System

### 1. Register a New Student / User
1. Open [http://localhost:5001](http://localhost:5001) and log in with `admin` / `admin123`.
2. Click **"Add New User"**.
3. Enter the student's **Name**, **Roll No**, and other required details.
4. The in-browser camera will open. Click **"Capture"** to take 20 face samples from slight angles.
5. The system will automatically train the LBPH recognition model.

### 2. Take Live Attendance
1. Click **"Take Attendance"** on the home dashboard (or navigate to `/detect`).
2. Allow browser camera access.
3. As students face the camera:
   - **Registered Student:** Green badge appears above the head: `🟢 Ram (95%)` and attendance is marked automatically.
   - **Unregistered / Unknown:** Yellow badge appears: `❓ Unknown Person`.
4. Click **"View Full Attendance"** to see the real-time attendance table.

### 3. View & Manage Database (phpMyAdmin UI)
1. Open [http://localhost:8080](http://localhost:8080).
2. Enter:
   - **Server:** `db`
   - **Username:** `root`
   - **Password:** *(leave blank)*
3. Select `digitalhajir` from the sidebar to view tables:
   - `users`: Registered students, roll numbers, and metadata.
   - `attendance`: Date, time, and marked records.
   - `admin`: Admin login credentials.

---

## 🛠️ Tech Stack & Architecture

- **Frontend**: HTML5, Vanilla CSS / Bootstrap, JavaScript, Canvas API
- **Face Detection (Client-side)**: [Pico.js](https://github.com/tehnokv/picojs) cascade detector (runs at 60 FPS in browser)
- **Face Recognition (Server-side)**: OpenCV LBPH (`cv2.face.LBPHFaceRecognizer`)
- **Backend API**: Python 3, Flask, Flask-Login
- **Database**: MySQL 8.0, PyMySQL / mysql-connector-python
- **DevOps**: Docker, Docker Compose, phpMyAdmin

---

## ❓ Troubleshooting & FAQs

- **Camera not opening in browser?**
  - Ensure you granted camera permissions to `http://localhost:5001` in your browser settings (look for the camera icon in the URL bar).
- **Port 5001 or 3306 already in use?**
  - Change the port in `docker-compose.yml` (e.g., `"5002:5000"`) or run with `PORT=5002 python app.py`.
- **Reset database or admin password?**
  - Run `python utils/create_admin.py` in your terminal to recreate or reset the admin password.