import cv2
from config.settings import *
from model.u2net_loader import load_model
from processing.segmentation import get_transform, get_mask
from camera.stream import get_rtsp_url, init_camera
from utils.performance import PerformanceTracker
from processing.target_mask import generate_mask
from processing.iou import calculate_iou
from ui.overlay import draw_iou

# model
model, device = load_model(MODEL_PATH)

transform = get_transform(INPUT_SIZE, MEAN, STD)

# camera
rtsp_url = get_rtsp_url(RTSP_USER, RTSP_PASSWORD, RTSP_HOST, RTSP_PORT, RTSP_PATH)

cap = init_camera(
    USE_RTSP,
    rtsp_url,
    WEBCAM_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    BUFFER_SIZE
)

tracker = PerformanceTracker()
frame_count = 0

target_mask = generate_mask("assets/shapes/dexter/dexter.png", model, device, transform)
target_mask = cv2.resize(target_mask, (FRAME_WIDTH, FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize frame to match target mask dimensions
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    mask = get_mask(frame, model, device, transform)

    iou_score = calculate_iou(mask, target_mask)

    draw_iou(mask, iou_score, position="top-right")
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    cv2.imshow("Target Mask", target_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()