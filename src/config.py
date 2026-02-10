import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "hand_gesture_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "gesture_model.h5")
PLOTS_PATH = os.path.join(BASE_DIR, "..", "reports", "plots")

# Gesture classes
GESTURE_CLASSES = ["ok","palm","point","rock","victory"]

#Gesture Actions 
GESTURE_ACTIONS = {
    "ok" : "open WhatsApp",
    "palm" : "open YouTube",
    "point" : "open Notepad",
    "rock" : "open Calculator",
    "victory" : "open Instagram"
}

# Camera Settings 
WCAM , HCAM = 648,488

# Prediction Interval
INTERVAL = 15