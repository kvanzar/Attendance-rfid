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

# --- System State Variables ---
system_mode = "WAITING" # WAITING, CAMERA_ON, RFID_ONLY, ENDED
latest_frame = None
class_start_time = None
class_end_time = None

# Tracks: { UID: {"name": str, "status": "IN", "punch_in": datetime, "last_in": datetime, "total_sec": float} }
student_records = {} 
rfid_to_name_map = {} 
name_to_rfid_map = {} # Reverse lookup for face scanning

def load_known_faces(folder_path="known_faces"):
    known_encodings, known_names = [], []
    global rfid_to_name_map, name_to_rfid_map
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return known_encodings, known_names

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            base_name = os.path.splitext(filename)[0]
            if "_" in base_name:
                name, uid = base_name.split("_", 1)
                uid = uid.upper()
                rfid_to_name_map[uid] = name
                name_to_rfid_map[name] = uid
            else:
                name = base_name
            
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
    return known_encodings, known_names

def export_final_excel():
    global class_start_time, class_end_time
    
    now = datetime.datetime.now()
    df_list = []
    
    # 1. ROW ONE: Master Start
    start_str = class_start_time.strftime("%Y-%m-%d %H:%M:%S") if class_start_time else "Unknown"
    df_list.append({
        "Student Name": ">>> MASTER START <<<",
        "Initial Punch In": start_str,
        "Total Time in Class": "",
        "RFID Number": "AE228B04",
        "Final Status": ""
    })
    
    # 2. MIDDLE ROWS: Student Data
    for uid, data in student_records.items():
        eff_time = data["total_sec"]
        if data["status"] == "IN":
            eff_time += (now - data["last_in"]).total_seconds()
            
        mins, secs = divmod(int(eff_time), 60)
        
        df_list.append({
            "Student Name": data["name"],
            "Initial Punch In": data["punch_in"].strftime("%H:%M:%S"),
            "Total Time in Class": f"{mins}m {secs}s",
            "RFID Number": uid,
            "Final Status": data["status"]
        })
        
    # 3. LAST ROW: Master End
    end_str = class_end_time.strftime("%Y-%m-%d %H:%M:%S") if class_end_time else now.strftime("%Y-%m-%d %H:%M:%S")
    df_list.append({
        "Student Name": ">>> CLASS ENDED <<<",
        "Initial Punch In": end_str,
        "Total Time in Class": "",
        "RFID Number": "AE228B04",
        "Final Status": ""
    })
    
    df = pd.DataFrame(df_list)
    df.to_excel("attendance_report.xlsx", index=False)
    print("✅ Final Attendance Excel Generated!")

def hardware_loop():
    global system_mode, student_records, latest_frame, class_start_time, class_end_time
    
    known_encodings, known_names = load_known_faces()
    MAC_SERIAL_PORT = '/dev/cu.usbserial-0001' # Update if needed
    
    try:
        ser = serial.Serial(MAC_SERIAL_PORT, 115200, timeout=1)
    except Exception:
        system_mode = "ERROR: ESP32 Not Found"
        return

    cap = None

    while True:
        # --- 1. HANDLE RFID SIGNALS ---
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"📥 Signal: {line}")
            
            if line == "MASTER_TAP":
                if system_mode == "WAITING":
                    print("Class Started! Camera ON.")
                    class_start_time = datetime.datetime.now()
                    system_mode = "CAMERA_ON"
                    student_records.clear()
                    cap = cv2.VideoCapture(0)
                    time.sleep(1)
                    
                elif system_mode == "CAMERA_ON":
                    print("Camera Stopped. RFID continuing seamlessly.")
                    system_mode = "RFID_ONLY"
                    if cap is not None:
                        cap.release()
                        cap = None
                        latest_frame = None

            elif line == "MASTER_HOLD":
                if system_mode in ["CAMERA_ON", "RFID_ONLY"]:
                    print("Class Ended by Master Hold > 3s.")
                    class_end_time = datetime.datetime.now()
                    system_mode = "ENDED"
                    if cap is not None:
                        cap.release()
                        cap = None
                        latest_frame = None
                    export_final_excel()

            elif line.startswith("STUDENT_TAP:"):
                # Ignore taps if class hasn't started or has ended
                if system_mode in ["CAMERA_ON", "RFID_ONLY"]:
                    uid = line.split(":")[1].strip()
                    
                    if uid in rfid_to_name_map:
                        name = rfid_to_name_map[uid]
                        
                        # Student must have punched in via face first!
                        if uid in student_records:
                            rec = student_records[uid]
                            now = datetime.datetime.now()
                            
                            if rec["status"] == "IN":
                                # Clock OUT
                                spent = (now - rec["last_in"]).total_seconds()
                                rec["total_sec"] += spent
                                rec["status"] = "OUT"
                                print(f"{name} Clocked OUT.")
                            else:
                                # Clock IN
                                rec["last_in"] = now
                                rec["status"] = "IN"
                                print(f"{name} Clocked IN.")
                        else:
                            print(f"❌ {name} must punch in with Face Scan first!")
                    else:
                        print("❌ Unregistered RFID Card.")

        # --- 2. HANDLE FACE RECOGNITION (Only if Camera is ON) ---
        if system_mode == "CAMERA_ON" and cap is not None:
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

                    color = (0, 0, 255) 
                    
                    if name != "Unknown":
                        uid = name_to_rfid_map.get(name)
                        
                        if uid:
                            now = datetime.datetime.now()
                            # Initial Face Punch-In Registration
                            if uid not in student_records:
                                student_records[uid] = {
                                    "name": name,
                                    "status": "IN",
                                    "punch_in": now,
                                    "last_in": now,
                                    "total_sec": 0
                                }
                                print(f"✅ {name} initially punched in via Face!")
                            
                            rec = student_records[uid]
                            if rec["status"] == "IN":
                                color = (0, 255, 0) # Green
                            else:
                                color = (0, 255, 255) # Yellow
                                
                    cv2.circle(frame, (center_x, center_y), radius, color, 3)
                    cv2.putText(frame, name, (center_x - radius, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                latest_frame = frame
        else:
            time.sleep(0.1)

# --- Flask Routes ---
@app.route('/')
def index(): 
    return render_template('index.html')

def generate_video_stream():
    global latest_frame, system_mode
    while True:
        if system_mode == "CAMERA_ON" and latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else: 
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed(): 
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    # Package attendees with their specific UI colors and messages
    attendees_ui = []
    for uid, data in student_records.items():
        if data["status"] == "IN":
            attendees_ui.append({"name": data["name"], "status": "IN", "msg": "Clocked In"})
        else:
            attendees_ui.append({"name": data["name"], "status": "OUT", "msg": "Clocked Out (Break)"})
            
    # Determine Header Text based on mode
    ui_state = "Waiting for Teacher"
    if system_mode == "CAMERA_ON": ui_state = "Class Active (Camera ON)"
    if system_mode == "RFID_ONLY": ui_state = "Class Active (Camera OFF, RFID ON)"
    if system_mode == "ENDED": ui_state = "Session Ended"
        
    return jsonify({
        "state": ui_state,
        "is_recording": system_mode == "CAMERA_ON",
        "attendees": attendees_ui, 
        "file_ready": os.path.exists("attendance_report.xlsx")
    })

@app.route('/download')
def download(): 
    return send_file("attendance_report.xlsx", as_attachment=True)

if __name__ == "__main__":
    threading.Thread(target=hardware_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=False)