import keras
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def build_model(input_size: int, n_nodes: int) -> keras.models.Model:
    input_layer = keras.layers.Input(shape=(input_size,))
    hidden_layer = keras.layers.Dense(n_nodes, activation="sigmoid")(input_layer)
    output_layer = keras.layers.Dense(1, activation="relu")(hidden_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


'''
 2.ii
'''
g_x = lambda _x: np.cos(2 * np.pi * _x) / (1 - (4 * _x) ** 2) * np.sin(np.pi * _x) / (np.pi * _x)
'''
2.iii
'''
h_x = lambda _x1, _x2: _x1 ** 2 + _x2 ** 2 + 2 * _x1 * _x2 * np.cos(np.pi * _x1 * _x2) + _x1 + _x2 - 1


def logic(x_1: np.array, x_2: np.array, x_3: np.array):
    y = list()
    for a, b, c in zip(x_1, x_2, x_3):
        y.append(int(a and b and c))

    return np.array(y)


def plot_function(x: np.array, y: np.array, fig_name: str) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")
    if x.ndim == 2:
        ax = pyplot.axes(projection='3d')
        ax.scatter(x[0], x[1], y, color="green", s=30)
    else:
        pyplot.plot(x, y)

    pyplot.show()


def main():
    x_1 = np.random.randint(0, 2, 50)
    x_2 = np.linspace(0, 4 * np.pi)
    x_3 = np.array([np.linspace(0, 1), np.linspace(0, 1)])

    y_1 = logic(x_1, x_1, x_1)
    y_2 = g_x(x_2)
    y_3 = h_x(x_3[0], x_3[1])
    plot_function(x_3, y_3, "h(x)")

    pass


if __name__ == '__main__':
    main()
