import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("frame", cv2.WINDOW_FULLSCREEN)

def calculate_3d_distance(lm1, lm2, W, H):
    x1, y1, z1 = int(lm1.x * W), int(lm1.y * H), lm1.z
    x2, y2, z2 = int(lm2.x * W), int(lm2.y * H), lm2.z

    z_scale = 1000

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + ((z2 - z1) * z_scale) ** 2)
    return distance

while cv2.getWindowProperty('frame', 0) >= 0:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture = "Nothing"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]

            thumb_index_dist = calculate_3d_distance(thumb_tip, index_tip, W, H)
            thumb_middle_dist = calculate_3d_distance(thumb_tip, middle_tip, W, H)

            threshold = 50  

            if thumb_index_dist < threshold and thumb_middle_dist < threshold:
                gesture = "Nothing"
            elif thumb_index_dist < threshold:  
                gesture = "Draw"
            elif thumb_middle_dist < threshold:  
                gesture = "Erase"

            x, y = int(thumb_tip.x * W), int(thumb_tip.y * H)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
