# Clash Detector Revit - YOLOv8

An automated computer vision solution that utilizes **YOLOv8** to detect structural and MEP (Mechanical, Electrical, and Plumbing) clashes within Autodesk Revit designs and visual exports.

---

## 📌 Features

* **Object Detection with YOLOv8:** Identifies overlapping or clashing elements from visual Revit exports.
* **Roboflow Integration:** Streamlined dataset management and automated model training via Roboflow API.
* **Custom Model Training:** Easily train custom detection models tailored to specific BIM element classes.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install ultralytics roboflow
