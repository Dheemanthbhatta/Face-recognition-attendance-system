# 📘 Face Recognition Attendance System

A real-time face recognition–based attendance management system built using **Python**, **OpenCV**, and **Flask**.  
This project allows teachers or administrators to register students, train a facial recognition model, capture attendance automatically, and view detailed attendance reports — all through a simple web interface.

---

## 🚀 Features

### 👤 Student Management
- Register new students with ID and name  
- Capture face images using webcam  
- Store images in dataset folder  
- View or delete registered students  

### 🤖 Face Recognition
- Uses **LBPH (Local Binary Patterns Histogram)** algorithm  
- Trains a model using captured student images  
- Recognizes faces in real time  
- Marks attendance automatically  

### 📝 Attendance Tracking
- Records date, time, student name, and class period  
- Stores attendance in `attendance.csv`  
- Prevents duplicate marking within the same session  

### 📊 Attendance Reports
- Period-wise attendance summary  
- Total classes held  
- Attendance percentage for each student  
- Clean tabular UI  

### 🌐 Web Interface (Flask)
- Home dashboard  
- Register student  
- Train model  
- Take attendance  
- View attendance records  
- List students  

---

## 🛠️ Tech Stack

| Component       | Technology         |
|----------------|--------------------|
| Backend         | Python, Flask      |
| Face Recognition| OpenCV (LBPH)      |
| Data Storage    | CSV, JSON          |
| Frontend        | HTML (Flask templates) |
| Camera          | Laptop/USB Webcam  |

---

## 📂 Project Structure

Face-Recognition-Attendance-System/ │── app.py │── teachers.py │── users.py │── README.md │── students.json │── attendance.csv │── trainer.yml │── dataset/ │── student_images/ │── attendance_images/

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Dheemanthbhatta/Face-recognition-attendance-system.git
cd Face-recognition-attendance-system

2️⃣ Install Dependencies
pip install opencv-contrib-python
pip install flask
pip install numpy

3️⃣ Run the Application
python app.py

4️⃣ Open in Browse
http://127.0.0.1:5000/



How It Works
The Face Recognition Attendance System follows a simple four‑step workflow:
1️⃣ Register Student
- The user enters the student’s name and ID through the web interface.
- The system activates the webcam and captures 20 face images of the student.
- Each image is cropped to the face region and saved in the dataset/ folder using the format : User.<student_id>.<image_number>.jpg
- The student’s ID–name mapping is stored in students.json.

2️⃣ Train the Model
- When the user clicks Train Model, the system:
- Loads all face images from the dataset/ folder
- Extracts labels (student IDs) from filenames
- Trains an LBPHFaceRecognizer model
- Saves the trained model as trainer.yml
- This model is later used to identify students during attendance.

3️⃣ Take Attendance
- The webcam opens and continuously scans for faces.
- Detected faces are passed to the trained LBPH model.
- If the model predicts a student ID with confidence < 70:
- The student is considered recognized
- Attendance is recorded in attendance.csv with:
- Date
- Student Name
- Time
- Selected Period
- If the face is not recognized, the system displays a “Student Not Registered” message.

4️⃣ View Attendance Records
- The system reads attendance.csv and organizes data:
- Period‑wise attendance
- Total classes held
- Number of classes attended by each student
- Attendance percentage
- Clean tables are generated using Flask templates for easy viewing.

