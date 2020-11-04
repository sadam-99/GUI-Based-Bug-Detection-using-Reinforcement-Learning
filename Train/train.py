from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Activation, Dense, BatchNormalization, GlobalAveragePooling2D, Flatten, Conv2D, Dropout
from keras.optimizers import SGD
import numpy as np
from build_dataset import DatasetBuilder
import cv2 as cv
import sys

def train():
    number_of_apps = 3000 # default

    if len(sys.argv) > 1:
        number_of_apps = int(sys.argv[1])

    train_data, train_label = get_data(number_of_apps)

    model = create_model()

    model.fit(train_data, train_label, validation_split=0.20, shuffle=True, batch_size=20, epochs=25)


def get_data(number_of_apps):
    return DatasetBuilder( n_apps=number_of_apps ).build()

def create_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(96, 54, 1)))

    model.add(Conv2D(filters=24, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    model.add(Conv2D(filters=36, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    model.add(Conv2D(filters=48, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    model.add(GlobalAveragePooling2D());

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(4))
    print(model.summary())

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer= sgd, loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    train()
