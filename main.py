import cv2
import numpy as np
from torchvision import transforms
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    a = np.array([[0, 1, 2],
                  [2, 2, 0]]).flatten()
    b = np.array([[0, 1, 0],
                  [1, 2, 0]]).flatten()
    c = zip(a, b)
    for x in c:
        print(x)