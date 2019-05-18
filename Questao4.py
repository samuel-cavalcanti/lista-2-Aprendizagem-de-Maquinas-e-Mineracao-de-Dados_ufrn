import keras
import matplotlib.pyplot as pyplot
from sklearn import metrics
import numpy as np
import cv2


def test_load_dataset():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    cv2.namedWindow("number", cv2.WINDOW_NORMAL)
    cv2.namedWindow("rotate number", cv2.WINDOW_NORMAL)
    cv2.imshow("number", train_images[0])
    rows, cols = train_images[0].shape

    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.uniform(0, 10), 1)
    new_image = cv2.warpAffine(train_images[0], rotation_matrix, (cols, rows))
    cv2.imshow("rotate number", new_image)
    cv2.waitKey(0)
    # pyplot.imshow(train_images[0])
    # pyplot.show()


def new_shape_for_keras(old_shape):
    new_shape = list(old_shape)
    new_shape.append(1)
    return tuple(new_shape)


def rotate_image(images: np.array) -> np.array:
    for i, image in enumerate(images):
        rows, cols = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.uniform(0, 10), 1)
        images[i] = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    return images


def pre_processing(train_images: np.array, train_labels: np.array, test_images: np.array, test_labels: np.array) \
        -> (np.array, np.array, np.array, np.array):
    x_train = rotate_image(train_images).reshape(new_shape_for_keras(train_images.shape))
    x_test = rotate_image(test_images).reshape(new_shape_for_keras(test_images.shape))
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


# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()

    # model = build_model(tuple([x_test.shape[1], x_test.shape[2], x_test.shape[-1]]), y_train.shape[-1])

    # history = model.fit(x_train, y_train, batch_size=50, epochs=3, validation_data=(x_test, y_test))
    # plot_history(history.history)
    model = keras.models.load_model("convNN")
    all_pred = model.predict(x_test)
    y_pred = [str(np.argmax(pred)) for pred in all_pred]
    y_true = [str(np.argmax(i)) for i in y_test]
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    print(matrix)
    # model.save("convNN")# val_acc = 0.973
