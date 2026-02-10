# src/app.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tkinter as tk
import time
from PIL import Image, ImageTk

from config import MODEL_PATH, GESTURE_CLASSES, GESTURE_ACTIONS, INTERVAL
from utils import perform_action

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")

label = tk.Label(root)
label.pack()

gesture_label = tk.Label(root, text="Gesture: ", font=("Arial", 14))
gesture_label.pack()

action_label = tk.Label(root, text="Action: ", font=("Arial", 14))
action_label.pack()

last_execution_time = time.time()

def detect_gesture():
    global last_execution_time
    current_time = time.time()

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)

        if current_time - last_execution_time >= INTERVAL:
            last_execution_time = current_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                    landmarks = np.array(landmarks).reshape(1, -1)

                    prediction = model.predict(landmarks, verbose=0)
                    gesture_idx = np.argmax(prediction)
                    gesture = GESTURE_CLASSES[gesture_idx]

                    gesture_label.config(text=f"Gesture: {gesture}")

                    if gesture in GESTURE_ACTIONS:
                        perform_action(GESTURE_ACTIONS[gesture], action_label)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.config(image=imgtk)

    root.after(10, detect_gesture)

detect_gesture()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
