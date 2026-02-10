import cv2 
import mediapipe as mp
import pandas as pd
from src.config import DATASET_PATH,WCAM,HCAM

mp_hands = mp.Solutions.hands
mp_draw = mp.Solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence =0.5)
cap = cv2.VideoCapture(0)
cap.set(3,WCAM)
cap.set(4,HCAM)

data =[]
labels =[]

gesture_name = input("Enter initial gesture name :")
print("\nPress 'n' to change gesture name, 'q' to quit.\n")

while True :
    ret, frame = cap.read()
    if not ret:
        break
    
    frame =cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            data.append(landmarks)
            labels.append(gesture_name)

    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Data", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        gesture_name = input("enter new gesture name:")
        
        
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(DATASET_PATH, index=False)

print(f"Dataset saved as {DATASET_PATH}")
    