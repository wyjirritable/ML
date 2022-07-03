import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from preprocessing import X_train, X_test, y_train, y_test
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=128, epochs=20)
# print(model.evaluate(X_test, y_test))

# Test the model
y_true = y_test.argmax(-1)
y_pred = model.predict(X_test).argmax(-1)
# generate confusion matrix

confusion_matrix(y_true, y_pred)
# calculate precision, recall, accuracy
print("Prec: " + str(precision_score(y_true, y_pred, average='weighted')))
print("Recall: " + str(recall_score(y_true, y_pred, average='weighted')))
print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
