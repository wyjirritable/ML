from keras.models import Sequential
from keras.layers import Dense, Flatten
from preprocessing import X_train, X_test, y_train, y_test
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(units=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=10)


# Test the model
y_true = y_test.argmax(-1)
y_pred = model.predict(X_test).argmax(-1)

# calculate precision, recall, accuracy
print("Prec: " + str(precision_score(y_true, y_pred, average='weighted')))
print("Recall: " + str(recall_score(y_true, y_pred, average='weighted')))
print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
