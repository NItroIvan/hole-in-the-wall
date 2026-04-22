def generate_mask(image_path, model, device, transform):
    import cv2
    from PIL import Image
    import numpy as np
    import torch

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = model(input_tensor)

    pred = d1[:, 0, :, :].squeeze().cpu().numpy()

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    
    mask = (pred * 255).astype(np.uint8)

    return mask