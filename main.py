import cv2
import time
import threading
from config.settings import *
from model.u2net_loader import load_model
from processing.segmentation import get_transform, get_mask
from camera.stream import get_rtsp_url, init_camera
from processing.target_mask import generate_mask
from processing.iou import calculate_iou
from ui.overlay import draw_iou, draw_progress_border

# model
model, device = load_model(MODEL_PATH)

transform = get_transform(INPUT_SIZE, MEAN, STD)

# camera
rtsp_url = None
if USE_RTSP:
    rtsp_url = get_rtsp_url(RTSP_USER, RTSP_PASSWORD, RTSP_HOST, RTSP_PORT, RTSP_PATH)

cap = init_camera(
    USE_RTSP,
    rtsp_url,
    WEBCAM_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    BUFFER_SIZE
)

target_mask = generate_mask("assets/shapes/dexter/dexter.png", model, device, transform)
target_mask = cv2.resize(target_mask, (FRAME_WIDTH, FRAME_HEIGHT))

interval = 2.5
border_duration = 10.0
iou_start_time = time.perf_counter()
border_start_time = time.perf_counter()

latest_frame = None
latest_mask = None
iou_score = 0.0

lock = threading.Lock()
stop_event = threading.Event()


def inference_worker():
    global latest_frame, latest_mask, iou_score, iou_start_time
    
    next_tick = time.perf_counter() + interval

    while not stop_event.is_set():
        with lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        mask = get_mask(frame, model, device, transform)

        current_time = time.perf_counter()

        with lock:
            if current_time >= next_tick:
                iou_score = calculate_iou(mask, target_mask)
                iou_start_time = current_time
                next_tick = current_time + interval
            latest_mask = mask

worker = threading.Thread(target=inference_worker)
worker.start()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    with lock:
        latest_frame = frame.copy()
        mask_to_show = latest_mask.copy() if latest_mask is not None else None
        current_iou = iou_score
        current_start_time = iou_start_time
    
    target_mask_to_show = target_mask.copy()

    target_mask_to_show = cv2.cvtColor(target_mask_to_show, cv2.COLOR_GRAY2BGR)
    draw_progress_border(target_mask_to_show, border_start_time, duration=border_duration)

    if time.perf_counter() - border_start_time >= border_duration:
        border_start_time = time.perf_counter()

    if mask_to_show is not None:
        draw_iou(target_mask_to_show, current_iou, current_start_time, position="top-right")
        cv2.imshow("Mask", mask_to_show)

    cv2.imshow("Target Mask", target_mask_to_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

stop_event.set()
worker.join()
cap.release()
cv2.destroyAllWindows()