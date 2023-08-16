"""
import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'Vitamin Pills Rotating Stock Video.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

class_name_dict = {0: 'alpaca'}

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
"""
"""
import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('.', 'videos')
image_files = ['image2.jpg', 'image1.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg']  # List the image files you want to process

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)

threshold = 0.5
class_name_dict = {0: 'capsules'}

for image_file in image_files:
    image_path = os.path.join(IMAGES_DIR, image_file)
    frame = cv2.imread(image_path)

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

"""
"""
import os
from ultralytics import YOLO
import cv2

IMAGE_DIR = os.path.join('.', 'videos')
image_file = 'image4.jpeg'  # Specify the image file you want to process

model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'last.pt')
model = YOLO(model_path)

threshold = 0.5
class_name_dict = {0: 'capsules'}  # Add the additional class and its corresponding class ID

image_path = os.path.join(IMAGE_DIR, image_file)
frame = cv2.imread(image_path)

results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
import os
from ultralytics import YOLO
import cv2

IMAGE_DIR = os.path.join('.', 'videos')
model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'last.pt')

threshold = 0.5
class_name_dict = {0: 'capsules'}  # Add the additional class and its corresponding class ID

model = YOLO(model_path)

# Get a list of all image files in the directory
image_files = [file for file in os.listdir(IMAGE_DIR) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    frame = cv2.imread(image_path)

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Create a named window and move it to a new position on the screen
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Image", 500, 200)

    cv2.imshow("Image", frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()


