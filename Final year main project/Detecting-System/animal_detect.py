# --- Python Script to Trigger Arduino from YOLOv8 ---

import serial
import time
from ultralytics import YOLO
import cv2

# 1️⃣ Connect to Arduino Nano
arduino = serial.Serial('COM8', 9600, timeout=1)  # Change COM4 to your actual port
time.sleep(2)
print("Arduino connected")

# 2️⃣ Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Replace 'best.pt' with your trained model path

# 3️⃣ Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4️⃣ Run YOLO inference
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # 5️⃣ Check for detected animals
    animal_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            # Replace with your animal names if needed
            if label in ['bear','bird','cat','dog','giraffe','horse','sheep','zebra']:
                animal_detected = True
                break

    # 6️⃣ Trigger Arduino
    if animal_detected:
        arduino.write(b'1')   # Turn ON LED
        print("Animal Detected – LED ON")
    else:
        arduino.write(b'0')   # Turn OFF LED
        print("No Animal – LED OFF")

    # Display the frame
    cv2.imshow("YOLOv8 Animal Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


