import serial
import time
from ultralytics import YOLO
import cv2

# Change this to the Bluetooth COM port you got after pairing ESP32
bt_com_port = 'COM4'  

arduino = serial.Serial(bt_com_port, 115200, timeout=1)
time.sleep(2)
print("Connected to ESP32 via Bluetooth")

model = YOLO('yolov8x.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)

    animal_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ['bear','bird','cat','dog','giraffe','horse','sheep','zebra']:
                animal_detected = True
                break

    if animal_detected:
        arduino.write(b'1')
        print("Animal Detected – LED ON")
    else:
        arduino.write(b'0')
        print("No Animal – LED OFF")

    cv2.imshow("YOLOv8 Animal Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()