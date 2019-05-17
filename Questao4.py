import keras
import matplotlib.pyplot as pyplot
import numpy as np


def test_load_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    pyplot.imshow(train_images[0])
    pyplot.show()


def new_shape_for_keras(old_shape):
    new_shape = list(old_shape)
    new_shape.append(1)
    return tuple(new_shape)


def pre_processing(train_images: np.array, train_labels: np.array, test_images: np.array, test_labels: np.array) \
        -> (np.array, np.array, np.array, np.array):
    x_train = train_images.reshape(new_shape_for_keras(train_images.shape))
    x_test = test_images.reshape(new_shape_for_keras(test_images.shape))
    y_train = keras.utils.to_categorical(train_labels)
    y_test = keras.utils.to_categorical(test_labels)

    return x_train, y_train, x_test, y_test


def load_dataset() -> (np.array, np.array, np.array, np.array):
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    return pre_processing(train_images, train_labels, test_images, test_labels)


def build_model(input_shape: tuple, output_size: int) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation=keras.activations.relu, input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation=keras.activations.relu))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(output_size, activation=keras.activations.softmax))
    model.compile(optimizer=keras.optimizers.adam(0.001), loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
    return model


def plot_training(label_train: str, label_val: str, title: str, y_label: str, train_data: list, val_data: list,
                  epochs: range):
    pyplot.figure(label_train)
    pyplot.plot(epochs, train_data, "bo", label=label_train)
    pyplot.plot(epochs, val_data, "r", label=label_val)
    pyplot.title(title)
    pyplot.xlabel("Epochs")
    pyplot.ylabel(y_label)
    pyplot.legend()
    pyplot.savefig("graficos/" + title, format="png")
    pyplot.show()


def plot_history(history) -> None:
    val_loss = history["val_loss"]
    val_acc = history["val_acc"]
    loss = history["loss"]
    acc = history["acc"]
    epochs = range(1, len(acc) + 1)

    plot_training("Training loss", "Validation loss", "Training and validation loss MNIST dataset", "Loss",
                  loss, val_loss, epochs)

    plot_training("Training acc", "Validation acc", "Training and validation accuracy MNIST dataset", "Accuracy",
                  acc, val_acc, epochs)

#TODO girar imagens e verifcar a classificação
#https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    model = build_model(tuple([x_test.shape[1], x_test.shape[2], x_test.shape[-1]]), y_train.shape[-1])
    history = model.fit(x_train, y_train, batch_size=50, epochs=3, validation_data=(x_test, y_test))
    plot_history(history.history)
    model.save("convNN")# val_acc = 0.973
