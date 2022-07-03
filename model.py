from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_score, recall_score, accuracy_score
from preprocessing import X_train, X_test, y_train, y_test


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())
model.add(Dense(units=512))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))


model.add(Dense(units=4, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint("model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto',
                      restore_best_weights=True)
callbacks_list = [checkpoint]

model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=1, validation_data=(X_test, y_test),
          callbacks=callbacks_list)

# Test the model
y_true = y_test.argmax(-1)
y_pred = model.predict(X_test).argmax(-1)


# calculate precision, recall, accuracy
print("Prec: " + str(precision_score(y_true, y_pred, average='weighted')))
print("Recall: " + str(recall_score(y_true, y_pred, average='weighted')))
print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
