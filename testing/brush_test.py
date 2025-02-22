import cv2
import numpy as np

# Create a blank image
image = np.ones((500, 500, 3), dtype=np.uint8) * 255
drawing = False  # True when mouse is pressed

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Press mouse
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:  # Move mouse while pressed
        if drawing:
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  # Draw blue circles

    elif event == cv2.EVENT_LBUTTONUP:  # Release mouse
        drawing = False

# Create window and set callback
cv2.namedWindow("Drag Drawing")
cv2.setMouseCallback("Drag Drawing", draw)

while True:
    cv2.imshow("Drag Drawing", image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
