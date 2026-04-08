# CLAUDE.md

## Project Overview

Learning project exploring **ExecuTorch** -- PyTorch's runtime for deploying ML models on edge devices (mobile, embedded, IoT). The focus is on exporting and optimizing pre-trained models (MobileNet V2, YOLO v8/v11) to the `.pte` format for constrained hardware.

## Tech Stack

- **Python:** 3.12
- **ML Framework:** PyTorch 2.x, TorchVision
- **Edge Runtime:** ExecuTorch 1.1
- **Object Detection:** Ultralytics (YOLOv8, YOLO11)
- **Model Formats:** PyTorch (.pt/.pth), ONNX (.onnx), ExecuTorch (.pte)
- **GPU:** CUDA 12.x + cuDNN (optional, for training)
- **Backend Optimization:** XNNPACK (ARM/x86 CPU), Core ML (iOS), Vulkan (Android GPU)

## Project Structure

```
executorch-learning/
├── export.py                  # MobileNet V2 export to ExecuTorch via XNNPACK
├── yolo_example.py            # YOLO11n export to ExecuTorch via Ultralytics
├── README.md                  # Documentation (Portuguese)
├── requirements.txt           # Python dependencies
├── model.pte                  # Exported MobileNet V2 model
├── fold_2_best.pth            # Custom trained model checkpoint
├── yolo11n.pt                 # YOLO11n pre-trained weights
├── yolov8n.pt                 # YOLOv8n pre-trained weights
├── *.onnx                     # ONNX exports (detection, classification, segmentation)
├── yolo11n_executorch_model/  # YOLO11n ExecuTorch artifacts + metadata
└── yolov8n_executorch_model/  # YOLOv8n ExecuTorch artifacts + metadata
```

## Export Pipeline

The standard workflow for all scripts follows this pattern:

1. Load a pre-trained model
2. Define sample inputs for shape inference
3. Export the computation graph (`torch.export.export()`)
4. Lower to a target backend (e.g., `XnnpackPartitioner`)
5. Serialize to `.pte` binary for edge deployment

## Running Scripts

```bash
# Activate virtual environment
source .venv/bin/activate

# Export MobileNet V2
python export.py

# Export YOLO11n
python yolo_example.py
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Git Conventions

- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`
- Single `main` branch
- Model weight files (`.pt`) are gitignored -- do not commit them
- ExecuTorch output directories (`*_executorch_model/`) are gitignored

## Notes

- README is written in Portuguese
- This is a learning/experimentation repo -- scripts are standalone and intentionally minimal
- YOLO metadata YAML files reference 80 COCO dataset classes with 640x640 input resolution
