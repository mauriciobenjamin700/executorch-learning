from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# https://docs.ultralytics.com/modes/export/#export-formats
model.export(
    format="executorch"
)
