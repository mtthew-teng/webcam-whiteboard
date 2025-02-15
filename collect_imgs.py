import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

num_classes = 2
dataset_size = 100

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for i in range(num_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))

    print('Collecting data for class {}'.format(i))

    done = False

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(counter)), frame)

            counter += 1

cap.release()
cv2.destroyAllWindows()