
import keras

ssl._create_default_https_context = ssl._create_unverified_context

# Load the Fashion MNIST dataset
data_set = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = data_set.load_data()

# Creating validation set
X_valid = X_train_full[:5000]
X_valid = X_valid / 255.0

# Validation labels
y_valid = y_train_full[:5000]

# Training set
X_train = X_train_full[5000:]
X_train = X_train / 255.0

# Training labels
y_train = y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Model architecture
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))  # Convert each input image into a 1D array
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation="softmax"))

# Model compilation
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Training
hist = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

print(model.evaluate(X_test, y_test))
