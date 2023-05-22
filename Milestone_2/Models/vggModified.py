import tensorflow as tf


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Resizing, Flatten, Dropout

def Vgg16Modified(input_shape, num_classes):
    model = tf.keras.models.Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform(),input_shape = input_shape
                                     ) )
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5)) 

    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5)) 




    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5)) 
    

    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5)) 


    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", use_bias=True, kernel_initializer=tf.keras.initializers.GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5)) 



    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model