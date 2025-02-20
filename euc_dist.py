import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("frame", cv2.WINDOW_FULLSCREEN)

gesture_buffer = deque(maxlen=10)
stable_gesture = "Nothing"

def calculate_3d_distance(landmark1, landmark2, W, H):
    x1, y1, z1 = int(landmark1.x * W), int(landmark1.y * H), landmark1.z
    x2, y2, z2 = int(landmark2.x * W), int(landmark2.y * H), landmark2.z
    z_scale = 1000
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + ((z2 - z1) * z_scale) ** 2)

while cv2.getWindowProperty('frame', 0) >= 0:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    new_gesture = "Nothing"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            thumb_index_dist = calculate_3d_distance(thumb_tip, index_tip, W, H)
            thumb_middle_dist = calculate_3d_distance(thumb_tip, middle_tip, W, H)

            threshold = 50

            if thumb_index_dist < threshold and thumb_middle_dist < threshold:
                new_gesture = "Nothing"
            elif thumb_index_dist < threshold:
                new_gesture = "Draw"
            elif thumb_middle_dist < threshold:
                new_gesture = "Erase"

            x, y = int(thumb_tip.x * W), int(thumb_tip.y * H)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    gesture_buffer.append(new_gesture)
    if gesture_buffer.count(new_gesture) > 7:
        stable_gesture = new_gesture

    cv2.putText(frame, f"Gesture: {stable_gesture}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
