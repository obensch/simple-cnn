import tensorflow as tf
from tensorflow import keras 

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

labels = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

x_train_n = x_train / 255
x_test_n = x_test / 255

x_valid, x_train = x_train_n[:5000], x_train_n[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
x_test = x_test_n

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)

tf.random.set_seed(77)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())	
model.add(keras.layers.Dense(128, activation="relu"))	
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

model_history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))

result = model.evaluate(x_test, y_test)
print('Test loss:', result[0])
print('Test accuracy:', result[1])
