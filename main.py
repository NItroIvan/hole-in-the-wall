import cv2
import time

from config.settings import *
from model.u2net_loader import load_model
from processing.segmentation import get_transform, get_mask
from camera.stream import get_rtsp_url, init_camera
from utils.performance import PerformanceTracker

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    mask = get_mask(frame, model, device, transform)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    elapsed = time.time() - start
    tracker.update(elapsed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"Avg time: {tracker.average():.3f}s")
cap.release()
cv2.destroyAllWindows()