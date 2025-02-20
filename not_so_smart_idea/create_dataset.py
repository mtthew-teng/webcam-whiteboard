import mediapipe as mp
import cv2
import os
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        print(f"Skipping: {dir_path} (Not a directory)")
        continue

    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)

        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping: {img_full_path} (Not an image)")
            continue

        data_aux = []
        img = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset saved as data.pickle")
