import numpy as np
import cv2

def process_mask(mask, threshold=127):
    # convert to binary mask (0 or 255)
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary

def calculate_iou(mask1, mask2):
    # ensure masks are binary
    mask1 = process_mask(mask1)
    mask2 = process_mask(mask2)

    # resize mask2 to match mask1's dimensions if they differ
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # convert to boolean arrays
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0
    
    iou = np.sum(intersection) / union_sum
    return iou