import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Draw', 1: 'Erase', 2: 'Nothing'}

cv2.namedWindow("frame", cv2.WINDOW_FULLSCREEN)

while cv2.getWindowProperty('frame', 0) >= 0:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = 255 * np.ones((H, W, 3), np.uint8)
    results = hands.process(frame_rgb)
    data_aux = []

    if results.multi_hand_landmarks:
        x_, y_ = [], []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # mp_drawing.draw_landmarks(
            #     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )

            hand_label = handedness.classification[0].label
            if hand_label == "Right":
                thumb_fingertip = hand_landmarks.landmark[4]
                x, y = int(thumb_fingertip.x * W), int(thumb_fingertip.y * H)
                z = thumb_fingertip.z

                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                # cv2.putText(frame, f"Index Tip: ({x}, {y}, {z:.3f})", (x + 20, y - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
                data_aux.extend([lm.x, lm.y])

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            print(predicted_character)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
