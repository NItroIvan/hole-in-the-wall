import cv2 
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

def get_transform(input_size, mean, std):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_mask(frame, model, device, transform):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = model(input_tensor)

        pred = d1[:, 0, :, :].squeeze().cpu().numpy()

        denom = pred.max() - pred.min()
        if denom > 1e-8:
            pred = (pred - pred.min()) / denom
        else:
            pred = np.zeros_like(pred)

        mask = (pred * 255).astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        return mask