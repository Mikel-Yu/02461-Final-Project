from ultralytics import YOLO
from PIL import Image

# Load a pretrained YOLOv8n model
model = YOLO('/Users/p/02.DTU/Project/yolov8/runs/detect/yolov8s_100epochs/weights/best.pt')

# Define path to directory containing images and videos for inference
source = '/Users/p/02.DTU/1.S/Intelligente systemer/chess/chess_data/test/images'

# Run inference on the source
results = model(source)  # generator of Results objects
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image