import cv2
from urllib.parse import quote

# build RTSP url
def get_rtsp_url(user, password, host, port, path):
    return(
        f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{path}"
    )

# init camera (RTSP or local camera)
def init_camera(use_rtsp, rtsp_url, webcam_index, width, height, buffer_size):
    if use_rtsp:
        cap = cv2.VideoCapture(rtsp_url)
    else:
        cap = cv2.VideoCapture(webcam_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    return cap

# read frame (helper)
def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

# release camera
def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
