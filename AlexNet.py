import tensorflow as tf

def built_model():
    model = tf.keras.models.Sequential(name="Alexnet")

    model.add(tf.keras.layers.Conv2D(filters=96,
                                     kernel_size=(11, 11),
                                     strides=4,
                                     input_shape=(507, 224, 3),
                                     activation='relu',
                                     padding="same",
                                     name="Layer1"))
    model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization_Layer1"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                        strides=(2, 2),
                                        name="MaxPool_Layer1"))

    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(5, 5),
                                     activation='relu',
                                     padding="same",
                                     name="Layer2"))
    model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization_Layer2"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                        strides=(2, 2),
                                        name="MaxPool_Layer2"))

    model.add(tf.keras.layers.Conv2D(filters=384,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding="same",
                                     name="Layer3"))

    model.add(tf.keras.layers.Conv2D(filters=256,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     padding="same",
                                     name="Layer4"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                        strides=(2, 2),
                                        name="MaxPool_Layer4"))

    model.add(tf.keras.layers.Flatten(name="Flatten"))

    model.add(tf.keras.layers.Dense(units=4096, name="Dense1"))
    model.add(tf.keras.layers.Dropout(0.5, name="Dropout_Dense1"))

    model.add(tf.keras.layers.Dense(units=4096, name="Dense2"))
    model.add(tf.keras.layers.Dropout(0.5, name="Dropout_Dense2"))

    model.add(tf.keras.layers.Dense(units=2, name="Output"))

    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005),
                  metrics=['accuracy'])

    return model

#model = built_model()
#print(model.summary())
