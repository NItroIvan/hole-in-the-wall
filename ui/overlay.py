import cv2

def draw_iou(frame, iou, position="top-left"):
    text = f"{iou * 100:2.1f}%"
    h, w = frame.shape[:2]

    if position == "top-left":
        pos = (10, 30)
    elif position == "top-right":
        pos = (w - 150, 30)
    elif position == "bottom-left":
        pos = (10, h - 10)
    else:
        pos = (w - 150, h - 10)

    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)