from flask import Flask, render_template, Response, jsonify, send_file
import threading
import cv2
import pandas as pd
import serial
import face_recognition
import os
import numpy as np
import datetime
import time

app = Flask(__name__)

# Global variables
system_state = "ARMED (Waiting for RFID)"
current_attendees = {}  # CHANGED: Now a dictionary to store {Name: Timestamp}
latest_frame = None
is_recording = False
face_match_counts = {}

def load_known_faces(folder_path="known_faces"):
    known_encodings, known_names = [], []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created '{folder_path}'. Please put photos inside.")
        return known_encodings, known_names

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
    return known_encodings, known_names

def hardware_loop():
    global system_state, current_attendees, latest_frame, is_recording, face_match_counts
    
    known_encodings, known_names = load_known_faces()
    MAC_SERIAL_PORT = '/dev/cu.usbserial-0001' # Update if your port changes
    
    try:
        ser = serial.Serial(MAC_SERIAL_PORT, 115200, timeout=1)
        print(f"Connected to ESP32 on {MAC_SERIAL_PORT}")
    except Exception as e:
        system_state = f"ERROR: ESP32 Not Found"
        return

    cap = None

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            if line == "START":
                is_recording = True
                system_state = "RECORDING (Session Active)"
                current_attendees.clear()
                face_match_counts.clear()
                cap = cv2.VideoCapture(0)
                time.sleep(1)

            elif line == "STOP":
                is_recording = False
                system_state = "ARMED (Waiting for RFID)"
                
                # Capture the exact time the RFID was tapped to close the session
                rfid_stop_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if cap is not None:
                    cap.release()
                    cap = None
                    latest_frame = None
                
                # Save data to Excel
                if current_attendees:
                    # Convert our dictionary to a DataFrame
                    df = pd.DataFrame(list(current_attendees.items()), columns=["Student Name", "Face Scan Time"])
                    
                    # Add the overall session end timestamp
                    df['Session Ended (RFID Tap)'] = rfid_stop_time
                    
                    df.to_excel("attendance_report.xlsx", index=False)

        if is_recording and cap is not None:
            ret, frame = cap.read()
            if ret:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]

                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    center_x = (left + right) // 2
                    center_y = (top + bottom) // 2
                    radius = int(max(right - left, bottom - top) / 1.5) 

                    color = (0, 0, 255) # RED
                    
                    if name != "Unknown":
                        if name in current_attendees:
                            color = (0, 255, 0) # GREEN
                        else:
                            face_match_counts[name] = face_match_counts.get(name, 0) + 1
                            if face_match_counts[name] >= 4:
                                # ADD TO DICT WITH CURRENT TIMESTAMP
                                current_attendees[name] = datetime.datetime.now().strftime("%H:%M:%S")
                                color = (0, 255, 0) # GREEN
                    
                    cv2.circle(frame, (center_x, center_y), radius, color, 3)
                    cv2.putText(frame, name, (center_x - radius, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                latest_frame = frame

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_stream():
    global latest_frame, is_recording
    while True:
        if is_recording and latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({
        "state": system_state,
        "is_recording": is_recording,
        # Extract just the names (keys) for the frontend list
        "attendees": list(current_attendees.keys()), 
        "file_ready": os.path.exists("attendance_report.xlsx")
    })

@app.route('/download')
def download():
    return send_file("attendance_report.xlsx", as_attachment=True)

if __name__ == "__main__":
    threading.Thread(target=hardware_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=False)