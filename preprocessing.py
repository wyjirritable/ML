import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_root = "data/"
labels = os.listdir(data_root)
X = []
y = []

for i in range(4):
    if labels[i][0] == '.':
        continue
    else:
        for file in os.listdir(data_root + labels[i]):
            if file[0] == '.':
                continue
            image = np.array(cv2.imread(data_root + labels[i] + '/' + file))
            # if image.shape != (256, 256, 3):
            #     image = image.repeat(4, axis=0)
            #     image = image.repeat(4, axis=1)
            # print(image.shape)
            # if image.shape != (224, 224, 3):
            # image.resize((128, 128, 3))
            #     print(image.shape)
                # print(image.shape)
            X.append(image)
            y.append(i)


X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                    np.array(y),
                                                    test_size=0.3)

y_train_ = y_train
y_test_ = y_test
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

