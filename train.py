# train.py

import torch
from src import config
from src.dataloader import create_resnet_dataloaders
from src.models import build_resnet_model, load_yolo_model
from src.engine import ResNetTrainer, YoloTrainer
from src.utils import get_class_names


def main():
    """
    Main function to orchestrate the training process based on the
    configuration file.
    """
    device = config.DEVICE
    print(f"[INFO] Using device: {device}")

    if config.MODEL_TYPE == "YOLOv8":
        print(f"[INFO] Initializing YOLOv8 training pipeline...")
        model = load_yolo_model()
        trainer = YoloTrainer(model, config.YOLO_CONFIG)

        print(f"[INFO] Starting YOLOv8 training...")
        trainer.train()
        print(
            f"\n[INFO] Training complete. Results saved in '{config.YOLO_CONFIG['project']}/'."
        )

    elif config.MODEL_TYPE == "ResNet":
        print(f"[INFO] Initializing ResNet training pipeline...")
        train_loader, val_loader, test_loader, class_names = create_resnet_dataloaders(
            config.RESNET_CONFIG["batch_size"], config.PROCESSED_DATA_PATH
        )

        print(f"[INFO] Building ResNeSt50 model...")
        model = build_resnet_model(
            num_classes=len(class_names), pretrained=config.RESNET_CONFIG["pretrained"]
        ).to(device)

        print(f"[INFO] Initializing ResNet Trainer...")
        trainer = ResNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            config=config.RESNET_CONFIG,
            device=device,
        )

        print(
            f"[INFO] Starting ResNet training for {config.RESNET_CONFIG['epochs']} epochs..."
        )
        trainer.train()
        print(f"\n[INFO] Training complete. Evaluating on test set...")
        trainer.evaluate()
        print(
            f"\n[INFO] ResNet pipeline finished. Model and plots saved in 'results/'."
        )

    else:
        raise ValueError(
            f"Unknown MODEL_TYPE in config.py: {config.MODEL_TYPE}. Choose 'YOLOv8' or 'ResNet'."
        )


if __name__ == "__main__":
    main()
