import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses 

def AlexNet(input_shape, num_classes):
    model=tf.keras.models.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4),activation='relu',input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(5,5), padding="same",activation="relu",padding='same',strides=(1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Conv2D(kernel_size=(3,3),filters=384,activation='relu',padding='same',strides=(1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384,kernel_size=(3,3),padding='same',activation='relu',strides=(1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3),padding='same',activation='relu',strides=(1,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes,activation='softmax'))
    return model

