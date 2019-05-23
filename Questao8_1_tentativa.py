import numpy as np
import keras
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os


def build_training_model(input_shape: tuple, nodes: int, output_size: int) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.Dense(nodes, activation=keras.activations.sigmoid, input_shape=input_shape))
    model.add(keras.layers.Dense(output_size, activation=keras.activations.sigmoid))
    model.compile(optimizer=keras.optimizers.rmsprop(), loss=keras.losses.mse, metrics=["mae"])

    return model


def show_image(name_window: str, image: np.array) -> None:
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_window, image)
    cv2.waitKey(0)


def plot_training(label_train: str, label_val: str, title: str, y_label: str, train_data: list, val_data: list,
                  epochs: range):
    pyplot.figure(label_train)
    pyplot.plot(epochs, train_data, "bo", label=label_train)
    pyplot.plot(epochs, val_data, "r", label=label_val)
    pyplot.title(title)
    pyplot.xlabel("Epochs")
    pyplot.ylabel(y_label)
    pyplot.legend()
    # pyplot.savefig("graficos/" + title, format="png")
    pyplot.show()


def plot_3d(vogal_pos: np.array, char_list: list, fig_name: str) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")

    ax = pyplot.axes(projection='3d')
    for x, y, z, char in zip(vogal_pos[:, 0], vogal_pos[:, 1], vogal_pos[:, 2], char_list):
        color = color_to_plot(char)
        ax.text(x, y, z, char, color=color)

    # ax.scatter(vogal_pos[:, 0], vogal_pos[:, 1], vogal_pos[:, 2], color="blue", label="Y_pred", s=30)
    pyplot.legend()

    # pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


'''
A = 0 I = 2
E = 1 O = 3
U = 4
'''


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


def color_to_plot(char: str) -> str:
    if char == "A":
        return "r"

    if char == "E":
        return "b"

    if char == "I":
        return "black"

    if char == "O":
        return "g"

    return "indigo"


def train_layer(input_shape: tuple, n_nodes: int, size: int, epochs: int,
                x: np.array, y: np.array, x_val: np.array, y_val: np.array,
                weights_file: str, new_data: str):
    model = build_training_model(input_shape, n_nodes, size)

    print("input shape: {} nodes: {} size {}".format(input_shape, n_nodes, size))

    history = model.fit(x, y, batch_size=30, epochs=epochs, validation_data=(x_val, y_val), verbose=False).history

    print("val loss: ", history["val_loss"][-1])
    # plot_training("Training loss", "Validation loss", "encoder loss", "Loss", history["loss"], history["val_loss"],
    #               range(1, n_epochs + 1))
    w_i = str(int(weights_file[13]) + 1)
    weights_file = weights_file.replace(weights_file[13], w_i)
    model.save_weights(weights_file)

    if n_nodes > 3:
        model.pop()
        data_train_for_second_layer = model.predict(x)
        data_test_for_second_layer = model.predict(x_val)

        d_i = str(int(new_data[13]) + 1)

        new_data = new_data.replace(new_data[13], d_i)

        np.savez(new_data, x_train=data_train_for_second_layer, x_test=data_test_for_second_layer)

        n_nodes //= 1.25
        train_layer(data_test_for_second_layer[0].shape, int(n_nodes), data_test_for_second_layer[0].size,
                    epochs, data_train_for_second_layer, data_train_for_second_layer, data_test_for_second_layer,
                    data_test_for_second_layer, weights_file, new_data)


def show_ypred(y_pred, y_true):
    y_pred = y_pred.reshape((y_pred.shape[0], 20, 20))
    y_true = y_true.reshape((y_true.shape[0], 20, 20))
    for pred, true in zip(y_pred, y_true):
        show_image("True", true)
        show_image("Pred", pred)


def load_models() -> (list, list):
    weights_file = "Redes_Salvas/1l"
    nodes = 400
    shape = tuple([400])
    size = 400
    encoded = list()
    decoded = list()
    for n in range(20):
        w_i = str(int(weights_file[13]) + 1)
        weights_file = weights_file.replace(weights_file[13], w_i)
        nodes //= 1.25
        model = build_training_model(shape, int(nodes), size)
        size = int(nodes)
        shape = tuple([size])
        model.load_weights(weights_file)
        encoded.append(model.layers[0])
        decoded.append(model.layers[1])

    return encoded, decoded


if __name__ == '__main__':
    file_name = "good_vowels.npz"

    x_test, x_train, y_test, y_train = np.load(file_name).values()
    x = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255
    val_x = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255
    n_nodes = int(x[0].size // 1.25)
    # n_epochs = 200
    # train_layer(x[0].shape, n_nodes, x[0].size, n_epochs, x, x, val_x, val_x, "Redes_Salvas/1l",
    #             "Redes_Salvas/2_layer_data")
    encoded, decoded = load_models()
    model = keras.Sequential()
    for layer in encoded:
        model.add(layer)

    # for layer in reversed(decoded):
    #     model.add(layer)


    chars = [type_to_char(n) for n in y_train]
    #
    y = model.predict(val_x)

    plot_3d(y, chars, "vogal pos")

    # show_ypred(y,val_x)