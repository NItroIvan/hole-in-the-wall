import cv2
import time

def draw_iou_text(frame, iou, x, y, w, h):
    text = f"{iou * 100:2.1f}%"

    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    text_x = x + (w - text_w) // 2
    text_y = y + (h + text_h) // 2

    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_progress_border(frame, start_time, duration=10.0, border_thickness=25):
    h, w = frame.shape[:2]

    elapsed = time.perf_counter() - start_time
    progress = min(elapsed / duration, 1.0)

    perimeter = 2 * (w + h)
    current_length = perimeter * progress

    cv2.rectangle(
        frame,
        (0, 0),
        (w - 1, h - 1),
        (0, 0, 255),
        thickness=border_thickness
    )

    remaining = current_length

    if remaining > 0:
        draw = min(remaining, w)
        cv2.line(frame, (0, 0), (int(draw), 0), (0, 255, 0), thickness=border_thickness)
        remaining -= draw
    
    if remaining > 0:
        draw = min(remaining, h)
        cv2.line(frame, (w - 1, 0), (w - 1, int(draw)), (0, 255, 0), thickness=border_thickness)
        remaining -= draw
    
    if remaining > 0:
        draw = min(remaining, w)
        cv2.line(frame, (w - 1, h - 1), (w - 1 - int(draw), h - 1), (0, 255, 0), thickness=border_thickness)
        remaining -= draw
    
    if remaining > 0:
        draw = min(remaining, h)
        cv2.line(frame, (0, h - 1), (0, h - 1 - int(draw)), (0, 255, 0), thickness=border_thickness)

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
    
    draw_iou_text(frame, iou, x, y, rect_w, rect_h)