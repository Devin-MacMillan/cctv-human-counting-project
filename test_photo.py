from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

results = model("/Users/tianhong/Documents/course/7240/project/a.jpg",imgsz=320)
# results[0].show()  # visualize prediction
