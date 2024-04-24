from ultralytics import YOLO
import cv2

import tkinter as tk
from threading import Thread

from matplotlib import figure
import matplotlib.image as image
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image

from shapely.geometry import Polygon

import chess

def order_points(pts):
    # 0: top-left
    # 1: top-right
    # 2: bottom-right
    # 3: bottom-left
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


def detect_corners(image):
    model_trained = YOLO("/Users/mikelyu/Desktop/Uni/02461 Intelligente Systemer/Final_Project/corner_detection_small_100epochs.pt") 
    results = model_trained.predict(source=image, line_width = 1, conf = 0.25, save_txt = True, save = True)
    
    boxes = results[0].boxes
    arr = boxes.xywh.numpy()
    points = arr[:,0:2]
    corners = order_points(points)
    
    return corners   



def four_point_transform(image, pts):
    img = Image.open(image)
    image = asarray(img)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # height and width - pythagorean theorem
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1]-br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype = "float32")
    
    # transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    img = Image.fromarray(warped, "RGB")
    
    if not isinstance(warped, np.ndarray):
        warped = np.asarray(warped)
    
    return img


def grid_on_chessboard(image):
    corners = np.array([[0, 0],
                        [image.size[0], 0],
                        [0, image.size[1]],
                        [image.size[0], image.size[1]]])
    
    corners = order_points(corners)
    
    plt.figure(figsize=(10, 10), dpi=80)
    
    implot = plt.imshow(image)
    
    TL = corners[0]
    TR = corners[1]
    BR = corners[2]
    BL = corners[3]
    
    def interpolate(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        dx = (x1 - x0) / 8
        dy = (y1 - y0) / 8
        pts = [(x0 + i * dx, y0 + i * dy) for i in range(9)]
        return pts
    
    ptsT = interpolate(TL, TR)
    ptsL = interpolate(TL, BL)
    ptsR = interpolate(TR, BR)
    ptsB = interpolate(BL, BR)
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')

    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL


def chess_pieces_detector(image):
    
    model_trained = YOLO("/Users/mikelyu/Desktop/Uni/02461 Intelligente Systemer/Final_Project/piece_detection_small.pt")
    results = model_trained.predict(source=image, line_width=1, conf=0.5, augment=False, save_txt=True, save=True)
    
    boxes = results[0].boxes
    detections = boxes.xyxy.numpy()
    return detections, boxes


def connect_square_to_detection(detections, square):
    # FEN-notation
    di = {0: 'b', 1: 'k', 2: 'n',
          3: 'p', 4: 'q', 5: 'r', 
          6: 'B', 7: 'K', 8: 'N',
          9: 'P', 10: 'Q', 11: 'R'}

    list_of_iou = []
    
    for i in detections:
        box = np.array([[i[0], i[1]], [i[2], i[1]], [i[2], i[3]], [i[0], i[3]]])
        
        # Adjust for high pieces if needed
        if box[3,1] - box[0,1] > 60:
            box[0,1] += 40
            box[1,1] += 40

        list_of_iou.append(calculate_iou(box, square))

    if not list_of_iou:  # Check if list_of_iou is empty
        return "empty"
    
    max_iou_index = list_of_iou.index(max(list_of_iou))
    max_iou_value = max(list_of_iou)
    
    num = max_iou_index

    piece = boxes.cls[num].tolist()

    if max_iou_value > 0.10:
        piece = boxes.cls[num].tolist()
        return di.get(piece, "empty")
    else:
        return "empty"



def create_fen(detections, all_squares):
    di = {0: 'b', 1: 'k', 2: 'n',
          3: 'p', 4: 'q', 5: 'r', 
          6: 'B', 7: 'K', 8: 'N',
          9: 'P', 10: 'Q', 11: 'R'}

    fen_rows = []

    for rank in all_squares:  # all_squares should be a 2D list of 8x8 squares
        fen_row = ""
        empty_count = 0

        for square in rank:
            piece = connect_square_to_detection(detections, square)
            
            if piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece

        if empty_count > 0:  # Handle trailing empty squares in a rank
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    fen_string = "/".join(fen_rows)
    return fen_string


def generate_squares(transformed_image):
    if not isinstance(transformed_image, np.ndarray):
        transformed_image = np.asarray(transformed_image)
        
    square_size = (transformed_image.shape[1] / 8, transformed_image.shape[0] / 8)
    all_squares = []

    for row in range(8):
        rank = []
        for col in range(8):
            top_left = (col * square_size[0], row * square_size[1])
            top_right = ((col + 1) * square_size[0], row * square_size[1])
            bottom_left = (col * square_size[0], (row + 1) * square_size[1])
            bottom_right = ((col + 1) * square_size[0], (row + 1) * square_size[1])
            square = [top_left, top_right, bottom_right, bottom_left]
            rank.append(square)
        all_squares.append(rank)
        
    
    for rank in all_squares:
        for square in rank:
            square_np = np.array(square, dtype = np.int32)
            cv2.polylines(transformed_image, [square_np], True, (0, 255, 0), 3)
            
    Image.fromarray(transformed_image).save("squares_on_transformed_image.jpg")


    return all_squares


def generate_board(detections, all_squares):
    fen_string = create_fen(detections, all_squares)
    board = chess.Board(fen_string)
    return board
    

def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Webcam - Press 's' to Save and Exit", frame)

        k = cv2.waitKey(1)
        if k % 256 == 115:  # Press 's' to save the image and exit
            img_name = "captured_chessboard.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} saved!")
            break

    cap.release()
    cv2.destroyAllWindows()
    return img_name

def start_capture():
    Thread(target=capture_image).start()



# GUI to trigger webcam capture
root = tk.Tk()
button = tk.Button(root, text="Capture Chessboard Image", command=start_capture)
button.pack()
root.mainloop()


# image = '/Users/mikelyu/Desktop/Uni/02461 Intelligente Systemer/Final_Project/Dataset/WIN_20240109_14_24_57_Pro.jpg'
image = "captured_chessboard.jpg"
corners = detect_corners(image)

transformed_image = four_point_transform(image, corners)

ptsT, ptsL = grid_on_chessboard(transformed_image)

detections, boxes = chess_pieces_detector(transformed_image)

all_squares = generate_squares(transformed_image)
board = generate_board(detections, all_squares)

print(board)
