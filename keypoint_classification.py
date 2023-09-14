import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, Dropout, Input, LSTM, Bidirectional, Embedding, TimeDistributed, ConvLSTM1D, GRU

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

from keras.regularizers import l2

from time import time



RANDOM_SEED = 42

dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'

NUM_CLASSES = 5

X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

print(np.shape(X_train))

print("Choose NN:")
print("1 - Dense")
print("2 - Conv 1D")
print("3 - Conv 1D + SVM")
print("4 - Conv 1D + LSTM") #
print("5 - ConvLSTM") #
print("6 - LSTM")
print("7 - Bi LSTM") #
print("8 - KNN\n")

opt = int(input("Enter Option: "))

if opt == 1:  # Dense NN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

elif opt == 2:  # CNN 1D
    X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], 1))

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(42, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))

elif opt == 3:  # CNN + SVM
    X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], 1))

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(42, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(NUM_CLASSES, kernel_regularizer=l2(0.01), activation='softmax'))

# elif opt == 4:  # CNN + LSTM
#     X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], 1))
#
#     model = Sequential()
#     model.add(
#         TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
#                         input_shape=(None, np.shape(X_train)[1], 1)))
#     model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
#     model.add(TimeDistributed(Dropout(0.5)))
#     model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
#     model.add(TimeDistributed(Flatten()))
#     model.add(LSTM(100))
#     model.add(Dropout(0.5))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(5, activation='softmax'))
#
# elif opt == 5:  # Conv LSTM 2D
#     X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], 1))
#
#     model = Sequential()
#     model.add(
#         ConvLSTM1D(filters=64, kernel_size=3, activation='relu', input_shape=(1, 42, 1)))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(5, activation='softmax'))

elif opt == 6:  # LSTM
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(42, 1)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

# elif opt == 7:  # Bi LSTM
#     X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1], 1))
#
#     model = Sequential()
#     model.add(Bidirectional(LSTM(256, return_sequences=True, input_shape=(42, 1))))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Dropout(0.5))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(5, activation='softmax'))

    model.build((None, 42, 1))

elif opt == 8: # KNN
    model = KNeighborsClassifier()
    model.fit(
        X_train,
        y_train,
    )
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    labels = sorted(list(set(y_test)))
    cmx_data = confusion_matrix(y_test, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_test)), 0)
    plt.show()
    quit()

model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)

es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = time()

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

print("CNN")
print(time()-start)

val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

model = tf.keras.models.load_model(model_save_path)

predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)

model.save(model_save_path, include_optimizer=False)

tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)

interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(np.shape(X_test))

if opt == 2 or opt == 3 or opt == 6 or opt == 7:
    X_test = np.reshape(X_test, (np.shape(X_test)[0], np.shape(X_test)[1], 1))

print(np.shape(X_test))

print("Accuracy Core: ")
print(accuracy_score(y_test, y_pred))
interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

# %%time
# Inference implementation
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
