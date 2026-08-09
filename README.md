# Structural-MEP Clash Detection using Computer Vision

A small end-to-end pipeline that detects structural-MEP clashes (e.g. a duct
passing through a column, beam, or slab) from 3D BIM screenshots, using a
custom Revit model as the data source and YOLOv8 for detection.

Clash detection is normally done manually or with expensive tools like
Autodesk Navisworks. This project explores whether a lightweight,
self-generated dataset and an object detector can approximate that workflow
at small scale.

## Pipeline

```
Autodesk Revit (structural model + MEP ductwork)
        │
        ├─ Interference Check → mathematically confirmed clash locations
        │
        ▼
Manual screenshot capture (clash + clean/no-clash views)
        │
        ▼
Roboflow (bounding-box annotation + augmentation)
        │
        ▼
YOLOv8 (Ultralytics, transfer-learned from COCO weights)
        │
        ▼
Evaluation (mAP50 / mAP50-95 / precision / recall)
```
## What was built

- Modeled a multi-story structural building (columns, beams, slabs) with MEP
  ductwork in Autodesk Revit.
- Used Revit's **Interference Check** tool to generate a ground-truth list of
  confirmed geometric clashes, rather than visually guessing overlaps.
- Prototyped view-duplication automation in **Dynamo** (visual programming)
  to batch-generate camera angles; used for part of the dataset before
  switching to manual screenshot capture for the remainder due to the
  complexity of scripting camera orientation.
- Captured and labeled 90 base images (45 clash / 45 clean) in Roboflow,
  expanded to ~230 images via augmentation (rotation, brightness, flips).
- Trained a YOLOv8n object detection model (transfer-learned from
  COCO-pretrained weights) on the resulting dataset.

## Results

| Metric | Value |
|---|---|
| mAP50 | 73.6% |
| mAP50-95 | 34.9% |

## Honest limitations

This is a proof-of-concept, not a production system:

- **Dataset size**: 90 base images is small for object detection (typical
  guidance is 500-1000+ images per class for a robust model). YOLOv8 was
  chosen partly *because* it is more data-efficient than two-stage detectors
  like Faster R-CNN, and benefits heavily from COCO-pretrained transfer
  learning — which is largely why a usable model emerged from this dataset
  size at all.
- **Single source**: all images come from one Revit model, so the dataset
  lacks geometric/architectural diversity. The model likely picked up on
  some model-specific visual quirks rather than fully general clash
  patterns.
- **The gap between mAP50 (73.6%) and mAP50-95 (34.9%)** indicates the model
  is reasonably good at *locating* clashes but not at drawing tight,
  precise bounding boxes — a common symptom of limited training data and
  box-shape variety.
- **No external validation**: training, validation, and test images all come
  from the same source model. The model has not been tested against a real
  project's BIM export or an unrelated building.
- **Dynamo automation is partial**: the view-duplication step works, but
  the camera-rotation and batch-export steps were not completed; most of
  the dataset was captured manually instead.

## Next steps

- Expand to multiple Revit models for geometric diversity.
- Finish the Dynamo pipeline for fully automated multi-angle export.
- Increase dataset size toward 500+ base images.
- Validate against a real-world or third-party BIM dataset.

## Tools used

Autodesk Revit, Dynamo, Roboflow, Python, Ultralytics YOLOv8, Google Colab.

## Repo contents

- `train_clash_detector.py` — training/evaluation script (also usable as a
  Colab notebook, cell-by-cell).
- `sample_images/` — a handful of example clash/clean screenshots (not the
  full dataset — see note below).

## About the dataset

The full image set is not included in this repo. It is hosted on Roboflow
(link below) since it's easier to browse, re-annotate, and version there
than as raw files in Git. A small sample of representative images is
included under `sample_images/` for a quick visual sense of the data.

**Roboflow project link:** [add your public/Universe link here]
