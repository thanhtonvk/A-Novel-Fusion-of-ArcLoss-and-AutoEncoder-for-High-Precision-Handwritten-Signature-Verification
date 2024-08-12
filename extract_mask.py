import numpy as np
import cv2
import os
for label in os.listdir('datasets/images'):
    os.makedirs(f'datasets/masks/{label}',exist_ok=True)
    for file_name in os.listdir(f'datasets/images/{label}'):
        path = f'datasets/images/{label}/{file_name}'
        image = cv2.imread(path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([90, 38, 0])
        upper = np.array([145, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cv2.imwrite(f'datasets/masks/{label}/{file_name}', mask)