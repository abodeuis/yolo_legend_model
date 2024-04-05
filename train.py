from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='yolo_train.yaml', epochs=100, imgsz=512)
results = model.val()

success = model.export(format='onnx')