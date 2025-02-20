import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("Right Index Fingertip Tracking", cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label

            if hand_label == "Right":
                index_fingertip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape

                x, y = int(index_fingertip.x * w), int(index_fingertip.y * h)
                z = index_fingertip.z

                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                cv2.putText(frame, f"Index Tip: ({x}, {y}, {z:.3f})", (x + 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("Right Index Fingertip Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
