from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano, small, medium, large, etc.)
model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.

# Train the model
model.train(
    data="/Users/tianhong/Documents/course/7240/project/HUMANDETECTION.v1i.yolov12/data.yaml",  # path to your YAML file
    epochs=5,
    imgsz=640,        # image size
    batch=16,         # adjust based on your system's memory
    device="cpu"      # or "0" for GPU
)

metrics = model.val()
results = model("/Users/tianhong/Documents/course/7240/project/a.jpg")
results[0].show()  # visualize prediction
