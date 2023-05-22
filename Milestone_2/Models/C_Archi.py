import tensorflow as tf

def B(input_shape, num_classes):
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Input(input_shape))
  model.add(tf.keras.layers.Rescaling(1./255))
  
  model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2,2)))
  model.add(tf.keras.layers.Dropout(0.1))

  model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", padding = "same")) 
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2,2)))
  model.add(tf.keras.layers.Dropout(0.1))

  model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,1), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2,2)))
  model.add(tf.keras.layers.Dropout(0.1))

  model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2,2)))
  model.add(tf.keras.layers.Dropout(0.1))

  model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(filters = 512, kernel_size = (1,1), activation = "relu", padding = "same"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2,2)))
  model.add(tf.keras.layers.Dropout(0.1))

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(units = 512, activation = "relu"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.2))

  model.add(tf.keras.layers.Dense(units = 100, activation = "softmax"))

  model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

  return model
