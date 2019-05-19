import numpy as np
from matplotlib import pyplot
import keras

# sinal discreto emitido pela fonte de sinal
s = lambda n: np.sin(0.075 * np.pi * n)


def V_1(n: int) -> float:
    if n >= 0:
        return -0.5 * V_1(n - 1) + np.random.normal()
    else:
        return 0


def V_2(n: int) -> float:
    if n >= 0:
        return 0.8 * V_2(n - 1) + np.random.normal()
    else:
        return 0


x = lambda n: s(n) + V_1(n)

y = lambda n: V_2(n) + 0.05 * s(n)


def plot_function(x: np.array, y: np.array, fig_name: str) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")

    pyplot.plot(x, y, "o", label="Y_pred")

    # pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


def plot_function_pred(y_pred: np.array, fig_name: str, y_true=None) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")

    pyplot.plot(y_pred, "b", label="Y_pred")
    if y_true is not None:
        pyplot.plot(y_true, "r", label="Y_true", markersize=1.5)
        pyplot.legend()

    pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


def generate_data(start: int, n_samples: int) -> (np.array, np.array, np.array):
    time = range(start, n_samples)

    font = [x(n) for n in time]

    noise = [y(n) for n in time]

    return np.array(time), np.array(font), np.array(noise)


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


def build_perceptron():
    perceptron = keras.models.Sequential()
    perceptron.add(keras.layers.Dense(1, input_shape=tuple([1])))

    perceptron.compile(optimizer=keras.optimizers.SGD(lr=0.001), loss=keras.losses.mse, metrics=[keras.metrics.mae])

    return perceptron


def build_multi_layer_perceptron():
    mlp = keras.models.Sequential()
    mlp.add(keras.layers.Dense(20, activation=keras.activations.sigmoid, input_shape=tuple([1])))
    mlp.add(keras.layers.Dense(1))
    mlp.compile(optimizer=keras.optimizers.rmsprop(), loss=keras.losses.mse, metrics=[keras.metrics.mae])

    return mlp


def evaluate_model(data_time: np.array, font_data: np.array, noise_data: np.array,
                   model: keras.Sequential, n_epochs: int, model_name: str) -> None:
    model.fit(data_time[0:5], noise_data[0:5], epochs=n_epochs)

    y_pred = model.predict(data_time)

    plot_function_pred(y_pred, fig_name=model_name + " noise", y_true=noise_data)
    plot_function_pred(font_data - y_pred.reshape(-1), model_name + " font function", s(data_time))


if __name__ == '__main__':
    data_time, font_data, noise_data = generate_data(0, 100)
    perceptron = build_perceptron()
    perceptron_epochs = 50
    multi_layer_perceptron = build_multi_layer_perceptron()
    mlp_epochs = 100

    evaluate_model(data_time, font_data, noise_data, perceptron, perceptron_epochs, "perceptron")
    evaluate_model(data_time, font_data, noise_data, multi_layer_perceptron, mlp_epochs, "multi layer perceptron")
