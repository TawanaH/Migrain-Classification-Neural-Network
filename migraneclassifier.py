import pandas as pd

dataset = pd.read_csv('data - data.csv')

x = dataset.drop(columns = ['Type'])

y = dataset['Type']

from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(23, input_shape=X_train.shape[1:], activation="relu"))
model.add(tf.keras.layers.Dense(14, activation="relu"))
model.add(tf.keras.layers.Dense(8, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 500, validation_data=(x_test, y_test))