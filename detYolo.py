# YOLO detection
from ultralytics import YOLO    # "pip3 install ultralytics" to  install ultralytics package
'''
Possible Improvements: Larger model size
Mosaic (mix images to imporove context)
Higher image resolution
Include partially visible people in training
Use negative samples in training
'''
class YoloDetecor:
    # init code

    # detection code

    # code to draw box

    # code from https://docs.ultralytics.com/tasks/detect/#how-do-i-train-a-yolo11-model-on-my-custom-dataset


    # Load a pretrained model
    model = YOLO("yolo11n.pt")

    # Train the model on your custom dataset        NEED TO CREATE .yaml FILE WITH DATASET
    model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)

    # Validate accuracy

    # Load the model
    model = YOLO("path/to/best.pt")

    # Validate the model
    metrics = model.val()
    print(metrics.box.map)  # mAP50-95
