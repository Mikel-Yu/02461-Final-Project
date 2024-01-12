from ultralytics import YOLO

from matplotlib.pyplot import figure
import matplotlib.image as image
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

import cv2

from shapely.geometry import Polygon
def order_points(pts):
    
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left
    
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# calculates iou between two polygons

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def detect_corners(image):
    
    # YOLO model trained to detect corners on a chessboard
    model_trained = YOLO("\desktop\intelligente systemer\corner_detection_small_100epochs.pt")
    results = model_trained.predict(source=image, line_thickness=1, conf=0.25, save_txt=True, save=True)

    # get the corners coordinates from the model
    boxes = results[0].boxes
    arr = boxes.xywh.numpy()
    points = arr[:,0:2]
    
    corners = order_points(points)
    
    return corners  

# perspective transforms an image with four given corners

def four_point_transform(image, pts):
      
    img = Image.open(image)
    image = asarray(img)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
   

    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    img = Image.fromarray(warped, "RGB")
    # img.show()    
    # return the warped image
    return img, maxHeight, maxWidth
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


def position_assigmnet(pieces,maxHeight,maxWidth):
    for i in range(len(pieces)): 
        if pieces[i][0] > maxHeight/2:
            if pieces[i][0] > maxHeight*0.75:
                if pieces[i][0] > maxHeight*0.875:
                    pieces[i].append(1)
                else:
                    pieces[i].append(2)
            else:
                if pieces[i][0] > maxHeight*0.625:
                    pieces[i].append(3)
                else:
                    pieces[i].append(4)
        else:
            if pieces[i][0] > maxHeight*0.25:
                if pieces[i][0] > maxHeight*0.375:
                    pieces[i].append(5)
                else:
                    pieces[i].append(6)
            else:
                if pieces[i][0] > maxHeight*0.125:
                    pieces[i].append(7)
                else:
                    pieces[i].append(8)


        if pieces[i][1] > maxWidth/2:
            if pieces[i][1] > maxWidth*0.75:
                if pieces[i][1] > maxWidth*0.875:
                    pieces[i].append("H")
                else:
                    pieces[i].append("G")
            else:
                if pieces[i][1] > maxWidth*0.625:
                    pieces[i].append("F")
                else:
                    pieces[i].append("E")
        else:
            if pieces[i][1] > maxWidth*0.25:
                if pieces[i][1] > maxWidth*0.375:
                    pieces[i].append("D")
                else:
                    pieces[i].append("C")
            else:
                if pieces[i][1] > maxWidth*0.125:
                    pieces[i].append("B")
                else:
                    pieces[i].append("A")

# definer billed
image = "C:\desktop\chess\WIN_20240109_14_29_08_Pro.jpg"

# model



model_pieces = YOLO("\desktop\intelligente systemer\piece_detection_small.pt")

corners = detect_corners(image)
img, maxHeight,maxWidth = four_point_transform(image,corners)
results_pieces = model_pieces(img)
# coordinates
a = []
pieces = []
corners = []
me = 10
size = 400
for r in results_pieces:
    boxes = r.boxes
    pieces_old = pieces
    pieces = []
    for box in boxes:
        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        # class name
        cls = int(box.cls[0])

        pieces.append([x1, y1, x2, y2,cls])
    print(pieces)
for i in range(len(pieces)):
    pieces[i] = [(pieces[i][0]+pieces[i][2])/2,pieces[i][3],pieces[i][-1]] 
different = position_check(pieces,pieces_old)
print(different)
if different == True:
    position_assigmnet(pieces)
print(pieces)
