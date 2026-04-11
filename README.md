# Shield Attendance: AI + RFID Secure Logging System

An end-to-end, hardware-integrated secure attendance system. This project uses an ESP32 and RC522 RFID module to trigger a local web application running on an Apple Silicon (M4) Mac. The Python backend streams live video via Flask, performs real-time facial recognition, and logs timestamped attendance data to an Excel report.

## Features

* **Hardware Triggered:** Session starts and ends via a physical RFID tap.
* **Real-Time Face Recognition:** Uses the `face_recognition` library (dlib) for fast, highly accurate face matching without needing to manually train a model.
* **Smart UI Feedback:** Live video feed with dynamic status rings:
    * 🔴 **Red:** Scanning for a face.
    * 🟢 **Green:** Face matched & attendance marked!
    * 🟡 **Yellow:** Duplicate scan prevention (Already marked).
* **Automated Reporting:** Generates a downloadable `.xlsx` file detailing the session start time, individual face scan times, and session end time.
* **Modern Dashboard:** Clean, responsive web UI built with Flask and Tailwind CSS.

---

## Hardware Setup

### Components Needed

* ESP32 Development Board
* RC522 RFID Reader
* Jumper Wires

### Wiring Diagram

Connect your RC522 module to the ESP32 using the following pins:

| RC522 Pin | ESP32 Pin       |
| :-------- | :-------------- |
| SDA (SS)  | D5              |
| SCK       | D18             |
| MOSI      | D23             |
| MISO      | D19             |
| IRQ       | *Not connected* |
| GND       | GND             |
| RST       | D22             |
| 3.3V      | 3.3V            |

---

## Software Installation

### ESP32 (Arduino IDE)

* Open the Arduino IDE and install the **ESP32** board manager (by Espressif).
* Go to **Sketch > Include Library > Manage Libraries** and install **MFRC522** by GithubCommunity.
* Upload the `esp32_rfid.ino` code to your board.
* Ensure you close the Arduino Serial Monitor after uploading so Python can access the port.

### Mac/PC Backend (Python)

* Ensure you have Python 3 installed. If you are on an Apple Silicon Mac, install CMake first (`brew install cmake`) before installing the Python packages.
* Open your terminal and install the required dependencies:
    ```bash
    pip install Flask opencv-python pandas pyserial face_recognition openpyxl "numpy<2"
    ```

---

## Project Structure

Ensure your project directory looks exactly like this before running:

```text
Shield_Attendance/
│
├── app.py                # Main Flask & hardware logic script
├── esp32_rfid.ino        # Code for the ESP32 microcontroller
├── known_faces/          # 📁 Put your student .jpg files here!
│   ├── Kshitij.jpg
│   └── Alice.png
│
└── templates/            # 📁 Required for Flask frontend
    └── index.html        # The Tailwind CSS frontend UI