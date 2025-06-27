# src/models.py

import torch.nn as nn
import timm
from ultralytics import YOLO
from src import config


def build_resnet_model(num_classes, pretrained=True):
    """
    Builds a ResNeSt50 model with a modified final layer for classification.
    """
    try:
        model = timm.create_model(
            config.RESNET_CONFIG["model_name"],
            pretrained=pretrained,
            num_classes=num_classes,
        )
        print(
            f"[INFO] Successfully loaded {config.RESNET_CONFIG['model_name']} from timm."
        )
    except Exception as e:
        print(
            f"[ERROR] Could not load {config.RESNET_CONFIG['model_name']}. Error: {e}"
        )
        print("[INFO] Falling back to torchvision's resnet50.")
        from torchvision import models

        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def load_yolo_model():
    """
    Loads a YOLOv8 model from the ultralytics library.
    """
    model_name = config.YOLO_CONFIG["model_name"]
    print(f"[INFO] Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    return model
