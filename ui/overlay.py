import cv2
import time

def draw_iou_text(frame, iou, x, y, w, h):
    text = f"{iou * 100:2.1f}%"

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    text_x = x + (w - text_w) // 2
    text_y = y + (h + text_h) // 2

    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_clock_fill(frame, x, y, w, h, start_time, duration=1.0):
    elapsed = (time.perf_counter() - start_time) % duration
    progress = elapsed / duration
    
    angle = int(progress * 360)

    center = (x + w // 2, y + h // 2)
    axes = (w // 2, h // 2)

    cv2.ellipse(frame, center, axes, 0, -90, -90 + angle, (128, 64, 128), -1)

def draw_iou(frame, iou, start_time, position="top-left"):
    h, w = frame.shape[:2]
    rect_w, rect_h = 150, 80

    if position == "top-left":
       x, y = 10, 10
    elif position == "top-right":
       x, y = w - rect_w - 10, 10
    elif position == "bottom-left":
         x, y = 10, h - rect_h - 10
    else:
        x, y = w - rect_w - 10, h - rect_h - 10
    
    draw_clock_fill(frame, x, y, rect_w, rect_h, start_time)
    draw_iou_text(frame, iou, x, y, rect_w, rect_h)