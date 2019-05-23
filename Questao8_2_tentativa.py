import numpy as np
import keras
from matplotlib import pyplot
import cv2
from sklearn import metrics


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


def plot_side_by_side(y_pred: np.array, y_noise: np.array, y_true: np.array, fig_name: str) -> None:
    y_pred, y_true = reshape_to_plot(y_pred, y_true)
    y_noise = y_noise.reshape((y_noise.shape[0], 20, 20))

    fig = pyplot.figure(fig_name, figsize=y_pred[0].shape)
    columns = 7
    rows = 3
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i <= columns:
            pyplot.imshow(y_true[i - 1])
        elif columns < i <= int(columns * 2):
            pyplot.imshow(y_noise[i - columns - 1])
        else:
            pyplot.imshow(y_pred[i - columns * 2 - 1])

    pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


def load_dataset(file_name: str) -> (np.array, np.array):
    x_test, x_train, y_test, y_train = np.load(file_name).values()
    x = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255
    x_val = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255
    return x, x_val, y_train, y_test


def build_autoencoder_model(input_shape: tuple, output_nodes: int) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=50, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(units=30, activation="relu"))
    model.add(keras.layers.Dense(units=50, activation="relu"))
    model.add(keras.layers.Dense(units=output_nodes, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.rmsprop(), loss='binary_crossentropy')

    return model


def show_image(name_window: str, image: np.array) -> None:
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_window, image)
    cv2.waitKey(-1)


def reshape_to_plot(y_pred: np.array, y_true: np.array) -> (np.array, np.array):
    return y_pred.reshape((y_pred.shape[0], 20, 20)), y_true.reshape((y_true.shape[0], 20, 20))


def show_ypred(y_pred: np.array, y_true: np.array):
    y_pred, y_true = reshape_to_plot(y_pred, y_true)
    for pred, true in zip(y_pred, y_true):
        show_image("True", true)
        show_image("Pred", pred)


def train_autoencoder(x: np.array, x_val: np.array, weights_file: str) -> None:
    model = build_autoencoder_model(x[0].shape, x[0].size)
    epochs = 70

    history = model.fit(x, x, epochs=epochs, batch_size=10, validation_data=(x_val, x_val), verbose=0).history
    plot_training("Training loss", "Validation loss", "deep encoder loss", "Loss", history["loss"], history["val_loss"],
                  range(1, epochs + 1))
    model.save_weights(weights_file)


def evalute_autoencoder(weights_file: str, x_val: np.array):
    model = build_autoencoder_model(x_val[0].shape, x_val[0].size)
    model.load_weights(weights_file)
    x_noise = x_val + np.random.normal(scale=0.5, size=x_val.shape)
    y_pred = model.predict(x_noise)
    plot_side_by_side(y_pred, x_noise, x_val, "Vogais")


def build_the_classifier(autoencoder_input_shape: tuple, autoencoder_output_size: int,
                         output_size_classifier: int, weights_file: str, ) -> keras.Sequential:
    model = build_autoencoder_model(autoencoder_input_shape, autoencoder_output_size)
    model.load_weights(weights_file)
    classifier = keras.Sequential()
    for layer in model.layers:
        layer.trainable = False
        classifier.add(layer)
    classifier.add(keras.layers.Reshape((20, 20, 1)))
    classifier.add(keras.layers.Conv2D(80, kernel_size=3, activation=keras.activations.relu))
    classifier.add(keras.layers.MaxPool2D())
    classifier.add(keras.layers.Conv2D(40, kernel_size=3, activation=keras.activations.relu))
    classifier.add(keras.layers.Flatten())
    classifier.add(keras.layers.Dense(output_size_classifier, activation=keras.activations.softmax))

    classifier.compile(optimizer=keras.optimizers.adam(0.001), loss=keras.losses.categorical_crossentropy,
                       metrics=["accuracy"])

    classifier.summary()

    return classifier


def train_classifier(x: np.array, x_test: np.array, y_train: np.array, y_test: np.array, weights_file: str):
    y_test = keras.utils.to_categorical(y_test, 5)
    y_train = keras.utils.to_categorical(y_train, 5)

    model = build_the_classifier(x[0].shape, x[0].size, y_train[0].size, weights_file)
    epochs = 200

    history = model.fit(x, y_train, batch_size=20, epochs=epochs, validation_data=(x_test, y_test)).history

    plot_training("Training loss", "Validation loss", "Vowels Classifier loss",
                  "Loss", history["loss"], history["val_loss"], range(1, epochs + 1))

    plot_training("Training acc", "Validation acc", "Vowels Classifier accuracy", "Accuracy",
                  history["acc"], history["val_acc"], range(1, epochs + 1))

    model.save("autoencoder_classifier")


def type_to_char(n: int) -> str:
    if n == 0:
        return "A"
    if n == 1:
        return "E"
    if n == 2:
        return "I"
    if n == 3:
        return "O"
    return "U"


def evaluete_classifier(model: keras.Sequential, x_test: np.array, y_test: np.array):
    all_pred = model.predict(x_test)
    y_pred = [type_to_char(int(np.argmax(pred))) for pred in all_pred]
    y_true = [type_to_char(int(np.argmax(i))) for i in y_test]
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=["A", "E", "I", "O", "U"])
    print(matrix)


'''
A = 0 I = 2
E = 1 O = 3
U = 4
'''

if __name__ == '__main__':
    x, x_val, y_train, y_test = load_dataset("Dataset_Questao8/good_vowels.npz")
    # train_autoencoder(x, x_val, "Autoencoder")

    train_classifier(x, x_val, y_train, y_test, "Redes_Salvas/Autoencoder")
    # model = keras.models.load_model("autoencoder_classifier")
    # evaluete_classifier(model, x_val, keras.utils.to_categorical(y_test, 5))
