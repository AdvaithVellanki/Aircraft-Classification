# src/utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from src import config


def get_class_names():
    """Returns a sorted list of class names."""
    # This ensures a consistent class order
    return sorted(config.CLASS_LABELS, key=config.CLASS_LABELS.get)


def plot_confusion_matrix(cm, class_names, save_path):
    """Plots and saves a confusion matrix."""
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(25, 25))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="BuGn")
    plt.title("ResNeSt50 Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.show()


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        # Propagate the score for the predicted class back to the model
        output[:, class_idx].backward(retain_graph=True)

        # Pool gradients across the width and height
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activation maps with the gradients
        activations = self.activations.detach()
        for i in range(len(pooled_gradients)):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activation maps
        heatmap = torch.mean(activations, dim=1).squeeze()

        # Apply ReLU to see only positive influences
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()


def visualize_grad_cam(image_path, heatmap):
    """Superimposes a heatmap on the original image and displays it."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Visualization")
    plt.axis("off")
    plt.show()
