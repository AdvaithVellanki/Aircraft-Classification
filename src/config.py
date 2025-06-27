# src/config.py

import torch

# --- MAIN CONFIGURATION ---
# Choose 'YOLOv8' for object detection or 'ResNet' for classification.
MODEL_TYPE = "YOLOv8"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KAGGLE_DATASET = "a2015003713/militaryaircraftdetectiondataset"
RAW_DATA_PATH = "data/Military Aircraft Detection Dataset"
PROCESSED_DATA_PATH = "data/processed"
RESULTS_PATH = "results"

# Class mapping (derived from the ResNet notebook)
CLASS_LABELS = {
    "A10": 0,
    "A400M": 1,
    "AG600": 2,
    "AV8B": 3,
    "B1": 4,
    "B2": 5,
    "B52": 6,
    "Be200": 7,
    "C130": 8,
    "C17": 9,
    "C5": 10,
    "E2": 11,
    "EF2000": 12,
    "F117": 13,
    "F14": 14,
    "F15": 15,
    "F16": 16,
    "F18": 17,
    "F22": 18,
    "F35": 19,
    "F4": 20,
    "J10": 44,
    "J20": 21,
    "JAS39": 22,
    "MQ9": 23,
    "Mig31": 24,
    "Mirage2000": 25,
    "P3": 42,
    "RQ4": 26,
    "Rafale": 27,
    "SR71": 28,
    "Su24": 41,
    "Su25": 40,
    "Su34": 29,
    "Su57": 30,
    "Tornado": 31,
    "Tu160": 32,
    "Tu95": 33,
    "U2": 34,
    "US2": 35,
    "V22": 36,
    "Vulcan": 37,
    "XB70": 38,
    "YF23": 39,
    "C2": 43,
    "E7": 45,
    "KC135": 46,
}

# --- YOLOv8 CONFIGURATION ---
YOLO_CONFIG = {
    "data_yaml": f"{PROCESSED_DATA_PATH}/data.yaml",
    "model_name": "yolov8m.pt",
    "epochs": 30,
    "imgsz": 640,
    "batch_size": 32,
    "lr0": 0.008987,
    "optimizer": "SGD",
    "name": "yolo_aircraft_run",
    "project": RESULTS_PATH,
}

# --- RESNET (ResNeSt50) CONFIGURATION ---
RESNET_CONFIG = {
    "model_name": "resnest50d.in1k",
    "pretrained": True,
    "num_classes": len(CLASS_LABELS),
    "image_size": 448,
    "batch_size": 16,
    "lr": 0.0005,
    "scheduler_factor": 0.15,
    "epochs": 10,
    "model_save_path": f"{RESULTS_PATH}/resnest_model.pt",
    "plot_save_path": f"{RESULTS_PATH}/resnet_confusion_matrix.png",
}
