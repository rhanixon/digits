import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# oad the data from MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
plt.imshow(x_train[0], cmap="gray")
plt.show()

# Normalize training and test data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build model
model = tf.keras.models.Sequential()
# Add the flatten layer
model.add(tf.keras.layers.Flatten())
# Build input and hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Build the output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Complile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(x=x_train, y=y_train, epochs=5)

# Evaluate model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
print('\nTest accuracy:', test_acc)

# Make Predictions
predictions = model.predict([x_test])
print(np.argmax(predictions[1000]))
plt.imshow(x_test[1000], cmap="gray")
plt.show()
