import numpy as np
import matplotlib.pyplot as pyplot
import keras

temp_serie = lambda n: np.sin(n + np.sin(n) ** 2)


def plot_function(y_pred: np.array, fig_name: str, y_true=None) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")

    pyplot.plot(y_pred, "o", label="Y_pred")
    if y_true is not None:
        pyplot.plot(y_true, "or", label="Y_true", markersize=1.5)
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


def generate_random_dataset(min: float, max: float, n_samples: int) -> (np.array, np.array):
    x = np.random.uniform(min, max, n_samples)
    return x, temp_serie(x)


def generate_dataset(min: float, max: float, n_samples: int, n_steps: int) -> (np.array, np.array):
    n = np.linspace(min, max, n_samples)

    x, y = list(), list()
    f_n = temp_serie(n)
    for i, value in enumerate(f_n):
        end = i + n_steps
        if end >= len(n):
            break
        seq_x = f_n[i:end]
        seq_y = f_n[end]
        x.append(seq_x)
        y.append(seq_y)

    return np.array(x).reshape((len(x), n_steps, 1)), np.array(y)


def build_model(input_shape: tuple, output_size: int) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(20, input_shape=input_shape))
    model.add(keras.layers.Dense(output_size))
    model.compile(optimizer="rmsprop", loss=keras.losses.mse, metrics=[keras.metrics.mae])

    return model


def evalute_narx():
    n_epochs = 20
    n_steps = 3
    x_train, y_train = generate_dataset(-2 * np.pi, 2 * np.pi, 500, n_steps)
    x_test, y_test = generate_dataset(-2 * np.pi, 2 * np.pi, 1500, n_steps)

    print(x_train.shape)
    model = build_model(tuple([n_steps, 1]), 1)

    history = model.fit(x_train, y_train, epochs=n_epochs, validation_data=(x_test, y_test)).history

    loss = history["loss"]
    val_loss = history["val_loss"]
    mean_absolute_error = history["mean_absolute_error"]
    val_mean_absolute_error = history["val_mean_absolute_error"]

    plot_training("Training loss", "Validation loss", "Temp series loss", "Loss", loss, val_loss,
                  range(1, n_epochs + 1))
    plot_training("Training MAE", "Validation MAE", "Temp series MAE", "MAE", mean_absolute_error,
                  val_mean_absolute_error,
                  range(1, n_epochs + 1))
    x_plot, y_plot = generate_dataset(-2 * np.pi, 2 * np.pi, 1000, n_steps)
    y_pred = model.predict(x_plot)

    plot_function(y_pred, "Temp series", y_plot)


if __name__ == '__main__':
    evalute_narx()
