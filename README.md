# Military Aircraft Classification using YOLOv8 and ResNeSt50

This project implements and evaluates two deep learning architectures for the detection and classification of military aircraft, using the "Military Aircraft Detection Dataset" from Kaggle. The primary goal is to compare the real-time object detection capabilities of YOLOv8 with the high-precision classification of a ResNeSt50 Convolutional Neural Network (CNN).

## Project Overview

As a popular hobby and a critical component of automated air traffic control and security, aircraft classification is a valuable application of AI. This repository provides a clear, modular, and configurable pipeline to train, evaluate, and compare two powerful models on this task, allowing users to easily switch between an object detection approach and a pure classification approach.

### Key Results

-   **YOLOv8m Model**: Achieved a **mean Average Precision (mAP50) of 0.83** and a precision of **0.87**. It excels at real-time detection but sometimes fails to detect small or distant aircraft against complex backgrounds.
-   **ResNeSt50 Model**: Achieved a superior classification **accuracy (precision) of 90%**. This architecture, which extends ResNet with split-attention blocks, proves highly effective for this fine-grained classification task.

## Features

-   **Dual Model Support**: Train and evaluate either **YOLOv8** (for detection) or a **ResNeSt50** CNN (for classification) from a single, unified codebase.
-   **Configurable Pipeline**: Easily switch between models and adjust hyperparameters like learning rate, batch size, and epochs in a central `config.py` file.
-   **Automated Data Setup**: Includes a script to automatically download the dataset from Kaggle, organize it into train/validation/test sets (80/10/10 split), and create the necessary label formats for both YOLO and ResNet.
-   **Model Evaluation**: Generates confusion matrices and key metrics (mAP, Precision, Recall, Accuracy) for comprehensive performance analysis.
-   **Grad-CAM Visualization**: For the ResNeSt model, you can generate Gradient-weighted Class Activation Maps (Grad-CAM) to visualize the regions of an image the model focuses on for its predictions.
-   **Inference Ready**: A simple script (`predict.py`) is provided to run predictions on new images using a trained model.

## Project Structure

This project is organized in a modular way to separate concerns and improve readability and maintenance.

```
/Aircraft_Classification/
|
|-- data/                     # Stores the raw and processed dataset (will be created when the code is run)
|-- results/                  # Directory for saved models and output plots (will be created when the code is run)
|-- src/                      # Contains all source code
|   |-- __init__.py
|   |-- config.py             # Main configuration file for model selection and hyperparameters
|   |-- data_setup.py         # Script to download and prepare the dataset
|   |-- dataloader.py         # Creates PyTorch DataLoaders for the ResNet model
|   |-- models.py             # Functions to build/load the ResNet and YOLO models
|   |-- engine.py             # Contains the Trainer classes with the core training/validation logic
|   |-- predict.py            # Script for running inference on new images
|   |-- utils.py              # Helper functions for plotting and class names
|
|-- train.py                  # Main script to execute the training pipeline
|-- requirements.txt          # All project dependencies
|-- README.md                 # This documentation file
```

### File Descriptions

-   **`train.py`**: The main entry point to start training. It reads the configuration and calls the appropriate modules.
-   **`src/config.py`**: A centralized file for all settings. **This is the main file you will edit** to switch between models (`YOLOv8` or `ResNet`) and adjust hyperparameters.
-   **`src/data_setup.py`**: A one-time setup script that handles downloading the dataset from Kaggle and processing it into the formats required by both YOLOv8 and ResNet.
-   **`src/dataloader.py`**: Contains the logic for creating efficient `DataLoader` instances for the ResNet model, including image transformations and augmentations.
-   **`src/models.py`**: Defines the model architectures. It has functions to load the YOLOv8 model from `ultralytics` and to build the ResNeSt50 model from `timm`.
-   **`src/engine.py`**: Contains the core training logic encapsulated in `YoloTrainer` and `ResNetTrainer` classes, separating the distinct training pipelines for each model.
-   **`src/predict.py`**: A script to perform inference. It can be run from the command line to classify a new image using a saved model.
-   **`src/utils.py`**: Holds helper functions, such as the logic for plotting a confusion matrix and generating Grad-CAM visualizations.
-   **`requirements.txt`**: A list of all Python packages required to run the project.

## Getting Started

### Prerequisites

-   Python 3.8+
-   `pip` and `virtualenv`
-   [Kaggle API Key](https://www.kaggle.com/docs/api) for automatic dataset download.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd Aircraft_Classification
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Setup

1.  **Set up your Kaggle API key:**
    -   Download your `kaggle.json` file from your Kaggle account settings.
    -   Place it in the required location: `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<Your-Username>\.kaggle\kaggle.json` (Windows).

2.  **Run the data setup script:**
    This one-time script will download the dataset, unzip it, split it into train/validation/test sets (80/10/10 split), and create the specific data formats required for both YOLO and ResNeSt.
    ```bash
    python src/data_setup.py
    ```

## Usage

### Configuration

All training settings can be modified in `src/config.py`. To choose which model to train, set the `MODEL_TYPE` variable:

```python
# in src/config.py
MODEL_TYPE = 'YOLOv8'  # Options: 'YOLOv8', 'ResNet'
```

### Training

Once the dataset is set up and the model is selected in the config file, start the training by running:

```bash
python train.py
```

Trained models and result plots (like confusion matrices) are saved in the `results/` directory.

### Prediction and Visualization

Use `predict.py` to run inference on a single image.

**Standard Prediction:**
```bash
# For YOLOv8
python src/predict.py --model_type YOLOv8 --model_path results/yolo_aircraft_run/weights/best.pt --image_path /path/to/image.jpg

# For ResNet
python src/predict.py --model_type ResNet --model_path results/resnest_model.pt --image_path /path/to/image.jpg
```

**Grad-CAM Visualization (ResNet Only):**

To generate a Grad-CAM heatmap overlay for the ResNet model, add the `--gradcam` flag. This helps visualize what parts of the image the model is focusing on.

```bash
python src/predict.py --model_type ResNet --model_path results/resnest_model.pt --image_path /path/to/image.jpg --gradcam
```
