from model import get_model
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.datasets.cifar10 import load_data
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
import os

save_dir = './saved_models'
batch_size = 64
epochs = 100

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

input_shape = x_train.shape[1:]
num_classes = 1 + y_test.max()
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)
model = get_model(input_shape, num_classes)
model_name = 'keras_mymodel.h5'




datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

step = 5
cnt = epochs // step
for i in range(cnt):
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=step, validation_data=(x_test, y_test), workers=7)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, "epoch_%d_" % (1+i) * (step) + model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
