import cv2
import numpy as np
from sort import Sort
from ultralytics import YOLO

# تحميل موديل YOLOv8 صغير
model = YOLO("yolov8n.pt")

# فتح الكاميرا
cap = cv2.VideoCapture(0)

# تهيئة التراكر
tracker = Sort()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    # الكشف باستخدام YOLO
    results = model(frame, verbose=False)[0]

    detections = []
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf[0])
        if conf > 0.3:  # فلترة بثقة أعلى من 0.3 (اختياري)
            detections.append([x1, y1, x2, y2, conf])

    if len(detections) > 0:
        dets = np.array(detections)
    else:
        dets = np.empty((0, 5))

    # تحديث التراكر
    tracks = tracker.update(dets)

    # رسم الصناديق والتراك على الإطار
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + SORT Tracking", frame)

    # خروج عند الضغط على زر ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC pressed, exiting...")
        break

cap.release()
cv2.destroyAllWindows()





