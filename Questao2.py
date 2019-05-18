import keras
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_function(x: np.array, y: np.array, fig_name: str, y_true=None) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")
    if x.ndim == 2:

        ax = pyplot.axes(projection='3d')
        if y_true is not None:
            ax.scatter(x[0], x[1], y_true, color="green", label="Y_true", s=10)
        ax.scatter(x[0], x[1], y, color="blue", label="Y_pred", s=30)
        pyplot.legend()

    else:
        pyplot.plot(x, y, "o", label="Y_pred")
        if y_true is not None:
            pyplot.plot(x, y_true, "or", label="Y_true", markersize=1.5)
            pyplot.legend()

    pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


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


def build_model_classifier(input_size: int, n_nodes: int) -> keras.models.Model:
    input_layer = keras.layers.Input(shape=(input_size,))
    first_hidden_layer = keras.layers.Dense(n_nodes, activation=keras.activations.relu)(input_layer)
    second_hidden_layer = keras.layers.Dense(n_nodes, activation=keras.activations.relu)(first_hidden_layer)
    output_layer = keras.layers.Dense(1, activation=keras.activations.sigmoid)(second_hidden_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001), loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
    return model


def logic(x_1: np.array):
    y = list()
    for a, b, c in x_1:
        y.append(int(a and b and c))

    return np.array(y)


def data_logic(n_samples: int) -> (np.array, np.array):
    x = np.array(
        [np.random.randint(0, 2, n_samples),
         np.random.randint(0, 2, n_samples),
         np.random.randint(0, 2, n_samples)]).T
    y = logic(x)

    return x, y


def pred_logic_function() -> None:
    n_epochs = 40
    x_train, y_train = data_logic(1000)
    x_test, y_test = data_logic(2500)

    model = build_model_classifier(3, 20)
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=20, validation_data=(x_test, y_test)).history

    acc = history["binary_accuracy"]
    val_acc = history["val_binary_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    plot_training("Training loss", "Validation loss", "Logic Function loss", "Loss", loss, val_loss,
                  range(1, n_epochs + 1))
    plot_training("Training acc", "Validation acc", "Logic Function accuracy", "Accuracy", acc, val_acc,
                  range(1, n_epochs + 1))


'''
 2.ii
'''
g_x = lambda _x: np.cos(2 * np.pi * _x) / (1 - (4 * _x) ** 2) * np.sin(np.pi * _x) / (np.pi * _x)


def g_x_data(n_samples: int) -> (np.array, np.array):
    x = np.random.uniform(1e-9, 4 * np.pi, n_samples)
    return x, g_x(x)


def build_model_regression(input_size: int, n_nodes: int) -> keras.models.Model:
    input_layer = keras.layers.Input(shape=(input_size,))
    first_hidden_layer = keras.layers.Dense(n_nodes, activation=keras.activations.relu)(input_layer)
    second_hidden_layer = keras.layers.Dense(n_nodes, activation=keras.activations.relu)(first_hidden_layer)
    output_layer = keras.layers.Dense(1)(second_hidden_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="rmsprop", loss=keras.losses.mse, metrics=[keras.metrics.mae])
    return model


def pred_g_x() -> None:
    n_epochs = 100
    x_train, y_train = g_x_data(1000)
    x_test, y_test = g_x_data(2500)

    model = build_model_regression(1, 20)
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=20, validation_data=(x_test, y_test)).history

    loss = history["loss"]
    val_loss = history["val_loss"]
    mean_absolute_error = history["mean_absolute_error"]
    val_mean_absolute_error = history["val_mean_absolute_error"]

    plot_training("Training loss", "Validation loss", "g(x) loss", "Loss", loss, val_loss,
                  range(1, n_epochs + 1))
    plot_training("Training MAE", "Validation MAE", "g(x) MAE", "MAE", mean_absolute_error,
                  val_mean_absolute_error,
                  range(1, n_epochs + 1))
    x_plot, y_plot = g_x_data(1000)
    y_pred = model.predict(x_plot)

    plot_function(x_plot, y_pred, "ModelPred_g(x)", y_pred)


'''
2.iii np.array([x_1, x_2]).shape
'''
h_x = lambda _x1, _x2: (_x1 ** 2) + (_x2 ** 2) + (2 * _x1 * _x2 * np.cos(np.pi * _x1 * _x2)) + _x1 + _x2 - 1


def h_x_data(n_samples: int) -> (np.array, np.array):
    x_1 = np.random.uniform(-1, 1, n_samples)
    x_2 = np.random.uniform(-1, 1, n_samples)

    return np.array([x_1, x_2]).T, h_x(x_1, x_2)


def pred_h_x() -> None:
    n_epochs = 1000
    x_train, y_train = h_x_data(1000)
    x_test, y_test = h_x_data(1000)

    model = build_model_regression(2, 20)
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=40, validation_data=(x_test, y_test)).history

    loss = history["loss"]
    val_loss = history["val_loss"]
    mean_absolute_error = history["mean_absolute_error"]
    val_mean_absolute_error = history["val_mean_absolute_error"]

    plot_training("Training loss", "Validation loss", "h(x) loss", "Loss", loss, val_loss,
                  range(1, n_epochs + 1))
    plot_training("Training MSE", "Validation MAE", "h(x) MAE", "MAE", mean_absolute_error,
                  val_mean_absolute_error,
                  range(1, n_epochs + 1))

    x_plot, y_plot = h_x_data(1000)
    y_pred = model.predict(x_plot)
    plot_function(x_plot.T, y_pred, "ModelPred_h(x)", y_plot)


if __name__ == '__main__':
    pred_logic_function()
    pred_g_x()
    pred_h_x()
