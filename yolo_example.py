from ultralytics import YOLO


model = YOLO("yolo11n.pt")

# https://docs.ultralytics.com/modes/export/#export-formats
model.export(
    format="executorch"
)
