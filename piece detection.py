from ultralytics import YOLO
import cv2
import math 
import numpy as np
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model_corners = YOLO("indsæt model for corners")
model_pieces = YOLO("indsæt model for brikkerne")
# object classes
classNames = []

#functions
def position_check(a,b):
    for i in range(len(a)):
        different = True
        for c in range(len(b)):
            if abs(a[i][0] - b[c][0]) > 100:
                print(1)
            elif abs(a[i][1] - b[c][1]) > 100:
                print(2)
            elif abs(a[i][2] - b[c][2]) > 100:
                print(3)
            elif abs(a[i][3] - b[c][3]) > 100:
                print(4)
            else:
                different = False
                break
        if different == True:
            return different


def position_assigmnet(pieces):
    for i in range(len(pieces)): 
        if pieces[i][0] > size/2:
            if pieces[i][0] > size*0.75:
                if pieces > size*0.875:
                    pieces[i].append(1)
                else:
                    pieces[i].append(2)
            else:
                if pieces[i][0] > size*0.625:
                    pieces[i].append(3)
                else:
                    pieces[i].append(4)
        else:
            if pieces[i][0] > size*0.25:
                if pieces > size*0.375:
                    pieces[i].append(5)
                else:
                    pieces[i].append(6)
            else:
                if pieces[i][0] > size*0.125:
                    pieces[i].append(7)
                else:
                    pieces[i].append(8)


        if pieces[i][1] > size/2:
            if pieces[i][1] > size*0.75:
                if pieces[i][1] > size*0.875:
                    pieces[i].append("H")
                else:
                    pieces[i].append("G")
            else:
                if pieces[i][1] > size*0.625:
                    pieces[i].append("F")
                else:
                    pieces[i].append("E")
        else:
            if pieces[i][1] > size*0.25:
                if pieces[i][1] > size*0.375:
                    pieces[i].append("D")
                else:
                    pieces[i].append("C")
            else:
                if pieces[i][1] > size*0.125:
                    pieces[i].append("B")
                else:
                    pieces[i].append("A")
# random variable
frame_counter = 0
a = []
pieces = []
corners = []
me = 10
size = 400
while True:
    success, img = cap.read()
    
    frame_counter += 1
    
    if frame_counter % 60 == 0: # Analyze every x'th frame
        results_corners = model_corners(img, stream=True)
        # coordinates
        for r in results_corners:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print(f"Confidence: {confidence}")

                # class name
                cls = int(box.cls[0])
                print(f"Class name: {classNames[cls]}")

                # Print coordinates and size
                print(f"Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                print(f"Width: {width}, Height: {height}")

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)

                corners.append([x1, y1, x2, y2,cls])
        corners.sort()
        if sum(corners[1]) < sum(corners[0]):
            l = corners[0]
            corners[0] = corners[1]
            corners[1] = l
        if sum(corners[3]) < sum(corners[2]):
            l = corners[3]
            corners[3] = corners[2]
            corners[2] = l
        pts1 = np.float32([corners[0], corners[1],
                       corners[2], corners[3]])
        pts2 = np.float32([[0, 0], [0, size],
                       [size,0], [size, size]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_img = cv2.warpPerspective(img, matrix, (500, 600))

    if frame_counter % 60 == 0: # Analyze every x'th frame
        results_pieces = model_pieces(transformed_img, stream=True)
        # coordinates
        for r in results_pieces:
            boxes = r.boxes
            pieces_old = pieces
            pieces = []
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print(f"Confidence: {confidence}")

                # class name
                cls = int(box.cls[0])
                print(f"Class name: {classNames[cls]}")

                # Print coordinates and size
                print(f"Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                print(f"Width: {width}, Height: {height}")

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)

                pieces.append([x1, y1, x2, y2,cls])
            for i in range(len(a)):
                pieces[i] = [(pieces[i][0]+pieces[i][2])/2,pieces[i][3],pieces[i][-1]] 
        different = position_check(pieces,pieces_old)
        if different == True:
            position_assigmnet(pieces):
    cv2.imshow('Webcam', img) 








cap.release()
cv2.destroyAllWindows()
