from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/Users/p/02.DTU/1.S/Intelligente systemer/chess/chess_data/data.yaml', epochs=150, imgsz=640, device='cpu')