import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Resizing

def resnet50(input_shape, num_classes):
    # Input tensor
    inputs = Input(shape=input_shape)

    # Stage 1

    x=Resizing(224, 224, interpolation="bilinear", input_shape=input_shape)(inputs)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Stage 2
    x = convolutional_block(x, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, [64, 64, 256])
    # x = identity_block(x, [64, 64, 256])

    # Stage 3
    x = convolutional_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    # x = identity_block(x, [128, 128, 512])
    # x = identity_block(x, [128, 128, 512])

    # Stage 4
    x = convolutional_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    # x = identity_block(x, [256, 256, 1024])
    # x = identity_block(x, [256, 256, 1024])
    # x = identity_block(x, [256, 256, 1024])
    # x = identity_block(x, [256, 256, 1024])

    # Stage 5
    x = convolutional_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    # x = identity_block(x, [512, 512, 2048])

    # Output layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    return model


def identity_block(input_tensor, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def convolutional_block(input_tensor, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, kernel_size=(1, 1), strides=strides, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, kernel_size=(1, 1), strides=strides, padding='valid')(input_tensor)
    shortcut = BatchNormalization()(shortcut)


    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x
    