import tensorflow.contrib.keras.api.keras as keras

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20


def get_model(input_shape, num_classes, lr=0.001):

    reg = 0.001
    leak = 0.2
    keep_prob = 0.25
    d1 = 128
    d2 = 256
    d3 = 128
    d4 = 64

    model = Sequential()
    model.add(Conv2D(d1, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(d1, (3, 3), kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(keras.layers.LeakyReLU(leak))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(d2, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(d2, (3, 3), kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(keras.layers.LeakyReLU(leak))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(d3, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(keras.layers.LeakyReLU(leak))
    model.add(Conv2D(d3, (3, 3), kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(keras.layers.LeakyReLU(leak))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(keep_prob))

    model.add(Conv2D(d4, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(keras.layers.LeakyReLU(leak))
    model.add(Conv2D(d4, (3, 3), kernel_regularizer=keras.regularizers.l2(reg)))
    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(keras.layers.LeakyReLU(leak))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(keep_prob))

    model.add(Flatten())
    """

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    """

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=lr)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    return model
