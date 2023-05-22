import tensorflow as tf


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Resizing, Flatten


def Vgg16(input_shape, num_classes):
    model=tf.keras.models.Sequential()



    model.add(Resizing(224, 224, interpolation="bilinear", input_shape=input_shape))




    model.add (Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))




    model.add (Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



    model.add (Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))




    model.add (Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



    model.add (Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add (Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same" ) )
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))




    model.add(Flatten())




    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))

    model.add(Dense(num_classes,activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model