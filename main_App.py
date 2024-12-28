import cv2
import mediapipe as mp
import time
import os

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Function to detect finger snapping
def is_snap(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    thumb_middle_distance = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
    return thumb_middle_distance < 0.05

# Main loop
last_processed_time = 0
frame_interval = 0.1  # Process every 100ms

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_processed_time < frame_interval:
        continue
    last_processed_time = current_time

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_snap(hand_landmarks):
                print("Snap detected! Shutting down the system...")
                os.system("shutdown /s /t 1")  # برای ویندوز
                # os.system("sudo shutdown now")  # برای لینوکس
                break

    cv2.imshow("Finger Snap Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()