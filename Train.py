"""
Structural-MEP Clash Detection using YOLOv8
---------------------------------------------
Pipeline: Revit (BIM model + Interference Check) -> Roboflow (annotation +
augmentation) -> YOLOv8 (Ultralytics) training -> evaluation on a held-out
validation set.

Run this in Google Colab (GPU runtime recommended: Runtime > Change runtime
type > T4 GPU) or any environment with a GPU.
"""

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
# !pip install ultralytics roboflow

from roboflow import Roboflow
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 2. Download the dataset from Roboflow
# ---------------------------------------------------------------------------
# Replace with your own API key (keep this private, do not commit it to
# GitHub -- see the README for how to handle this safely).
ROBOFLOW_API_KEY = "YOUR_API_KEY"
WORKSPACE = "your-workspace-id"
PROJECT = "structural-mep-clash-detection"
VERSION = 1

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download("yolov8")

DATA_YAML = f"{dataset.location}/data.yaml"

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
# yolov8n (nano) is chosen deliberately: it is far more data-efficient than
# two-stage detectors (e.g. Faster R-CNN) and starts from COCO-pretrained
# weights, which matters a lot given the small dataset size here (~230
# images post-augmentation).
model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/detect",
    name="clash_detector",
)

# ---------------------------------------------------------------------------
# 4. Evaluate on the validation set
# ---------------------------------------------------------------------------
metrics = model.val()

print("\n--- Evaluation metrics ---")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

# ---------------------------------------------------------------------------
# 5. Run inference on a new image
# ---------------------------------------------------------------------------
# Example (uncomment and set your own path):
# test_results = model("path/to/test_image.jpg")
# test_results[0].show()
# test_results[0].save(filename="prediction_result.jpg")

# In Google Colab, to upload an image interactively instead of hardcoding a
# path:
#
# from google.colab import files
# uploaded = files.upload()
# test_image_path = list(uploaded.keys())[0]
# test_results = model(test_image_path)
# test_results[0].show()

# ---------------------------------------------------------------------------
# 6. Locate best weights (useful if you restart the runtime later)
# ---------------------------------------------------------------------------
# !find runs -name "best.pt"
# model = YOLO("runs/detect/clash_detector/weights/best.pt")