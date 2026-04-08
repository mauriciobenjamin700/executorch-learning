from pathlib import Path

import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner
)
from executorch.exir import to_edge_transform_and_lower
import onnxruntime.tools.convert_onnx_models_to_ort as ort_convert


img_size = int(input("Image size (ex: 224, 384): "))
num_classes = int(input("Number of classes: "))
file_name: str = input("File name (ex: efficientnet_v2s.pth): ")

model = models.efficientnet_v2_s()
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,  # type: ignore
    num_classes
)

state_dict = torch.load(file_name, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

sample_inputs = (torch.randn(1, 3, img_size, img_size),)

# ExecuTorch (.pte)
et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("efficientnet_v2s.pte", "wb") as f:
    f.write(et_program.buffer)

print(f"Exportado: efficientnet_v2s.pte ({img_size}x{img_size})")

# ONNX (.onnx)
onnx_path = Path("efficientnet_v2s.onnx")

torch.onnx.export(
    model,
    sample_inputs[0],  # type: ignore
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

print(f"Exportado: {onnx_path}")

# ORT (.ort)
output_dir = Path(".")

ort_convert.convert_onnx_models_to_ort(
    model_path_or_dir=onnx_path,
    output_dir=output_dir,
    optimization_styles=[ort_convert.OptimizationStyle.Fixed],
)

print(f"Exportado: efficientnet_v2s.ort em {output_dir.resolve()}")
