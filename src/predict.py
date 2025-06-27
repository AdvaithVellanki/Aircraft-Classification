# src/predict.py

import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.models import build_resnet_model
from src import config
from src.utils import get_class_names, GradCAM, visualize_grad_cam


def predict(args):
    """
    Runs inference on a single image using the specified model and optionally
    generates a Grad-CAM visualization for ResNet models.
    """
    device = config.DEVICE
    class_names = get_class_names()

    if args.model_type.lower() == "resnet":
        print(f"[INFO] Running inference with ResNet model: {args.model_path}")

        # Load Model
        model = build_resnet_model(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()

        # Image Transformations
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        config.RESNET_CONFIG["image_size"],
                        config.RESNET_CONFIG["image_size"],
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image = Image.open(args.image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)

        prediction = class_names[preds.item()]
        print(f"\n---> ResNet Prediction: {prediction}")

        # Grad-CAM Visualization
        if args.gradcam:
            print("[INFO] Generating Grad-CAM visualization...")
            # Note: You may need to change 'layer4' to the correct final conv block name
            # depending on the exact model architecture from timm.
            target_layer = model.layer4
            grad_cam = GradCAM(model, target_layer)

            heatmap = grad_cam(image_tensor)
            visualize_grad_cam(args.image_path, heatmap)

    elif args.model_type.lower() == "yolo":
        print(f"[INFO] Running inference with YOLOv8 model: {args.model_path}")
        model = YOLO(args.model_path)
        results = model(args.image_path)

        print("\n---> Detection Results:")
        for result in results:
            result.show()  # Display the image with bounding boxes
            print(result.boxes)

    else:
        raise ValueError("Invalid model type. Choose 'ResNet' or 'YOLOv8'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aircraft Classification Inference Script"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["ResNet", "YOLOv8"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights (.pt).",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM visualization (ResNet only).",
    )

    args = parser.parse_args()
    predict(args)
