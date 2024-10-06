import os
import numpy as np
import torch


def visualize_label(image, label):
    original_image_rgb = np.stack((image,) * 3, axis=-1)
    colors = np.array([(63, 142, 247), (252, 246, 232), (234, 101, 140)]) / 255.0
    overlay = np.zeros_like(original_image_rgb)
    # Overlay each channel with the assigned color
    overlay[label == 1] = colors[0]  # Use colors[0] for class label 1
    alpha = 0.9  # Blend factor
    result = (1 - alpha) * original_image_rgb + alpha * overlay
    result = np.clip(result, 0, 1)
    return result


def visualize_prediction(args, image, model):
    checkpoint_path = os.path.join(args.model_path, args.model_name + "_checkpoint.pt")
    checkpoint = torch.load(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image = image.unsqueeze(0).unsqueeze(0).to(device)
    pred = model(image)
    pred = torch.softmax(pred, dim=1)
    pred_label = torch.argmax(pred, dim=1)
    image = image.squeeze().squeeze().cpu()
    pred_label = pred_label.squeeze().cpu()
    result = visualize_label(image, pred_label)
    return result

