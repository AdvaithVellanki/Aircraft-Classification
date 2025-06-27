# src/predict.py

import torch
from torchvision import transforms
from PIL import Image
import argparse
import cv2
from ultralytics import YOLO

from src.models import build_resnet_model
from src import config
from src.utils import get_class_names


def predict(model_type, model_path, image_path):
    """
    Runs inference on a single image using the specified model.
    """
    device = config.DEVICE
    class_names = get_class_names()

    if model_type.lower() == "resnet":
        print(f"[INFO] Running inference with ResNeSt model: {model_path}")
        model = build_resnet_model(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

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

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)

        prediction = class_names[preds.item()]
        print(f"\n---> Prediction: {prediction}")

    elif model_type.lower() == "yolo":
        print(f"[INFO] Running inference with YOLOv8 model: {model_path}")
        model = YOLO(model_path)
        results = model(image_path)

        print("\n---> Detection Results:")
        results[0].show()  # Display the image with bounding boxes
        print(results[0].boxes)

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
        help="Type of model to use for inference.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model weights (.pt file).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image for prediction.",
    )

    args = parser.parse_args()
    predict(args.model_type, args.model_path, args.image_path)
