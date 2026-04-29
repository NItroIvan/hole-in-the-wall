import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# RTSP / CAMERA
RTSP_USER = os.getenv("RTSP_USER", "admin")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD")
RTSP_HOST = os.getenv("RTSP_HOST")
RTSP_PORT = os.getenv("RTSP_PORT", "554")
RTSP_PATH = os.getenv("RTSP_PATH", "h264Preview_01_main")

# VIDEO SETTINGS
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SKIP = 6
BUFFER_SIZE = 1

USE_RTSP = True
WEBCAM_INDEX = 0

# MODEL
MODEL_PATH = "saved_models/u2netp/u2netp.pth"
INPUT_SIZE = 256

# NORMALIZATION
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# PERFORMANCE
MEASUREMENT_TIME = 10
