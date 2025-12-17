import cv2
import os
import numpy as np
from flask import Flask, render_template_string, request, redirect
from datetime import datetime
import csv
import json
from collections import defaultdict

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_dir = 'dataset'
model_file = 'trainer.yml'
attendance_file = 'attendance.csv'
students_file = 'students.json'

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Load student name mapping
def load_students():
    if os.path.exists(students_file):
        with open(students_file, 'r') as f:
            return json.load(f)
    return {}

# Save student name mapping
def save_student(student_id, name):
    students = load_students()
    students[str(student_id)] = name
    with open(students_file, 'w') as f:
        json.dump(students, f)

def get_images_and_labels():
    faces = []
    ids = []
    for image_name in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, image_name)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        id = int(image_name.split('.')[1])
        faces.append(gray)
        ids.append(id)
    return faces, np.array(ids)

def train_model():
    faces, ids = get_images_and_labels()
    if len(faces) > 0:
        recognizer.train(faces, ids)
        recognizer.save(model_file)

def mark_attendance(name, period):
    now = datetime.now()
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now.strftime("%Y-%m-%d"), name, now.strftime("%H:%M:%S"), period])

@app.route('/')
def home():
    return render_template_string("""
        <div style='text-align:center;'>
            <h2>Face Recognition Attendance</h2>
            <a href="/register">Register Student</a><br><br>
            <a href="/train">Train Model</a><br><br>
            <a href="/attendance">Take Attendance</a><br><br>
            <a href="/records">View Records</a><br><br>
        </div>
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        student_id = int(request.form['id'])
        save_student(student_id, name)

        cam = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y + h, x:x + w]
                cv2.imwrite(f"{dataset_dir}/User.{student_id}.{count}.jpg", face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Registering - Press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
                break
        cam.release()
        cv2.destroyAllWindows()
        return redirect('/')
    return render_template_string("""
        <div style='text-align:center;'>
            <h3>Register Student</h3>
            <form method="POST">
                Name: <input name="name" required><br>
                Student ID (numeric): <input name="id" required type="number"><br><br>
                <button type="submit">Register</button>
            </form>
        </div>
    """)

@app.route('/train')
def train():
    train_model()
    return "<div style='text-align:center;'><h3>Training Complete</h3><a href='/'>Back</a></div>"

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        period = request.form['period']
        if not os.path.exists(model_file):
            return "<p>Model not trained. Please register and train first.</p>"

        students = load_students()
        recognizer.read(model_file)
        cam = cv2.VideoCapture(0)
        recognized = None
        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                id_, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if conf < 70:
                    name = students.get(str(id_), f"User_{id_}")
                    recognized = name
                    mark_attendance(name, period)
                    break
            cv2.imshow('Attendance - Press Q to capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or recognized:
                break
        cam.release()
        cv2.destroyAllWindows()
        return render_template_string("""
            <div style='text-align:center;'>
                <h3>Attendance Result</h3>
                <p>Name: {{ name }}</p>
                <p>Period: {{ period }}</p>
                <a href="/">Back</a>
            </div>
        """, name=recognized or "Unknown", period=period)

    return render_template_string("""
        <div style='text-align:center;'>
            <h3>Mark Attendance</h3>
            <form method="POST">
                Select Period:
                <select name="period">
                    <option value="Period 1">Period 1</option>
                    <option value="Period 2">Period 2</option>
                    <option value="Period 3">Period 3</option>
                    <option value="Period 4">Period 4</option>
                    <option value="Period 5">Period 5</option>
                    <option value="Period 6">Period 6</option>
                </select><br><br>
                <button type="submit">Capture</button>
            </form>
        </div>
    """)

@app.route('/records')
def records():
    rows = []
    period_wise_data = defaultdict(list)
    period_class_dates = defaultdict(set)
    period_attendance_count = defaultdict(lambda: defaultdict(int))
    students = load_students()

    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                date, name, time, period = row
                rows.append(row)
                period_wise_data[period].append(row)
                period_class_dates[period].add(date)
                period_attendance_count[period][name] += 1

    return render_template_string("""
        <div style='text-align:center;'>
            <h3>Attendance Records</h3>

            {% for period, period_rows in period_wise_data.items() %}
                <h4>{{ period }}</h4>
                <p>Total Classes Held: {{ period_class_dates[period]|length }}</p>
                <table border="1" style='margin:auto;'>
                    <tr><th>Date</th><th>Name</th><th>Time</th></tr>
                    {% for r in period_rows %}
                        <tr>
                            <td>{{ r[0] }}</td><td>{{ r[1] }}</td><td>{{ r[2] }}</td>
                        </tr>
                    {% endfor %}
                </table>
                <h5>Summary for {{ period }}</h5>
                <table border="1" style='margin:auto;'>
                    <tr><th>Name</th><th>Classes Attended</th><th>Attendance %</th></tr>
                    {% for name in students.values() %}
                        {% set attended = period_attendance_count[period].get(name, 0) %}
                        {% set total = period_class_dates[period]|length %}
                        <tr>
                            <td>{{ name }}</td>
                            <td>{{ attended }}</td>
                            <td>{{ (attended / total * 100) | round(2) if total else 0 }}%</td>
                        </tr>
                    {% endfor %}
                </table><br>
            {% endfor %}
            <a href="/">Back</a>
        </div>
    """, period_wise_data=period_wise_data, period_attendance_count=period_attendance_count,
         period_class_dates=period_class_dates, students=students)

if __name__ == '__main__':
    app.run(debug=True)
