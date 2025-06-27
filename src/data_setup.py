# src/data_setup.py

import os
import shutil
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kaggle
from src import config


class DataSetup:
    def __init__(self):
        self.raw_data_path = config.RAW_DATA_PATH
        self.processed_path = config.PROCESSED_DATA_PATH
        self.annotations_csv = os.path.join(self.raw_data_path, "_annotations.csv")

    def download_dataset(self):
        """Downloads and unzips the dataset from Kaggle if not present."""
        if not os.path.exists(self.raw_data_path):
            print("[INFO] Downloading dataset from Kaggle...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                config.KAGGLE_DATASET, path="data", unzip=True
            )
            print("[INFO] Dataset downloaded and unzipped.")
        else:
            print("[INFO] Raw dataset already exists. Skipping download.")

    def _prepare_dataframe(self):
        """Reads annotations and prepares the main dataframe with class IDs."""
        df = pd.read_csv(self.annotations_csv)
        class_names = sorted(df["name"].unique())
        class_to_id = {name: i for i, name in enumerate(class_names)}
        df["class_id"] = df["name"].map(class_to_id)
        return df, class_names, class_to_id

    def setup(self):
        """Orchestrates the full data setup process."""
        self.download_dataset()

        if os.path.exists(self.processed_path):
            print(
                f"[INFO] Processed data directory '{self.processed_path}' already exists. Skipping setup."
            )
            return

        print("[INFO] Preparing data for YOLOv8 and ResNet...")
        df, class_names, class_to_id = self._prepare_dataframe()

        # Split data into train, val, test
        image_files = df["filename"].unique()
        train_val_files, test_files = train_test_split(
            image_files, test_size=0.1, random_state=42
        )
        train_files, val_files = train_test_split(
            train_val_files, test_size=(1 / 9), random_state=42
        )  # 1/9 of 90% is 10%

        splits = {"train": train_files, "val": val_files, "test": test_files}

        # Create directory structures and move files
        for split, files in splits.items():
            # Directories for YOLO and ResNet
            yolo_img_dir = os.path.join(self.processed_path, "yolo", "images", split)
            yolo_lbl_dir = os.path.join(self.processed_path, "yolo", "labels", split)
            resnet_split_dir = os.path.join(self.processed_path, "resnet", split)

            os.makedirs(yolo_img_dir, exist_ok=True)
            os.makedirs(yolo_lbl_dir, exist_ok=True)
            os.makedirs(resnet_split_dir, exist_ok=True)

            for class_name in class_names:
                os.makedirs(os.path.join(resnet_split_dir, class_name), exist_ok=True)

            print(f"[INFO] Processing and copying {split} files...")
            for filename in tqdm(files, desc=f"Processing {split} set"):
                src_img_path = os.path.join(self.raw_data_path, "JPEGImages", filename)
                if not os.path.exists(src_img_path):
                    continue

                # Copy image for YOLO
                shutil.copy(src_img_path, yolo_img_dir)

                # Copy image into class folder for ResNet
                img_class = df[df["filename"] == filename]["name"].iloc[0]
                shutil.copy(src_img_path, os.path.join(resnet_split_dir, img_class))

                # Create YOLO label file for this image
                self._create_yolo_label(df, filename, yolo_lbl_dir)

        # Create YOLO data.yaml
        self._create_yaml_file(class_names)
        print("\n[INFO] Dataset setup complete.")

    def _create_yolo_label(self, df, filename, label_dir):
        """Creates a single YOLO format label file."""
        img_df = df[df["filename"] == filename]
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        img_w = img_df["width"].iloc[0]
        img_h = img_df["height"].iloc[0]

        with open(label_path, "w") as f:
            for _, row in img_df.iterrows():
                x_center = (row["xmin"] + row["xmax"]) / 2 / img_w
                y_center = (row["ymin"] + row["ymax"]) / 2 / img_h
                width = (row["xmax"] - row["xmin"]) / img_w
                height = (row["ymax"] - row["ymin"]) / img_h
                f.write(f"{row['class_id']} {x_center} {y_center} {width} {height}\n")

    def _create_yaml_file(self, class_names):
        """Creates the data.yaml file for YOLOv8 training."""
        yaml_path = os.path.join(self.processed_path, "data.yaml")
        yaml_content = {
            "train": os.path.abspath(
                os.path.join(self.processed_path, "yolo", "images", "train")
            ),
            "val": os.path.abspath(
                os.path.join(self.processed_path, "yolo", "images", "val")
            ),
            "test": os.path.abspath(
                os.path.join(self.processed_path, "yolo", "images", "test")
            ),
            "nc": len(class_names),
            "names": class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)
        print(f"\n[INFO] YOLO data.yaml created at {yaml_path}")


if __name__ == "__main__":
    data_setup = DataSetup()
    data_setup.setup()
