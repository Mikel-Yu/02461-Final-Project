import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("Dog.mp4")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes ):
        (x, y, x2, y2) = bbox
        
        cv2.rectangle(frame, (x, y), (x2, y2), (225, 0, 225), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (225, 0, 225), 2)
    
    print(bboxes)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()