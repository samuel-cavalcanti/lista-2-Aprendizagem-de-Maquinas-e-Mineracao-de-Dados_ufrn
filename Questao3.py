import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import keras
from sklearn import svm as SVM
from sklearn import metrics


def plot_function(x: np.array, y: np.array, fig_name: str, y_true=None) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")

    blue = True
    red = True

    if y_true is not None:
        print(y_true)
        for px, py, color in zip(x, y, y_true):
            if color == "or" and red:
                pyplot.plot(px, py, color, label="Class 1", markersize=1.5)
                red = False
            if color == "ob" and blue:
                pyplot.plot(px, py, color, label="Class 0", markersize=1.5)
                blue = False
            pyplot.plot(px, py, color, markersize=1.5)

        pyplot.legend()
        # pyplot.plot(x, y, y_true, markersize=1.5)
    # pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


def class_to_color(y) -> np.array:
    new_target = list()
    for target in y:
        if target:
            new_target.append("or")
        else:
            new_target.append("ob")

    return np.array(new_target)


def get_class(x: float, y: float) -> int:
    if x >= 0 and y >= 0:
        if np.sqrt(x ** 2 + (y - 1) ** 2) <= 1 and np.sqrt((x - 1) ** 2 + y ** 2) <= 1:
            type_class = 1
        else:
            type_class = 0

    elif x >= 0 > y:
        if np.sqrt(x ** 2 + (y + 1) ** 2) <= 1 and np.sqrt((x - 1) ** 2 + y ** 2) <= 1:
            type_class = 1
        else:
            type_class = 0

    elif x < 0 <= y:
        if np.sqrt(x ** 2 + (y - 1) ** 2) <= 1 and np.sqrt((x + 1) ** 2 + y ** 2) <= 1:
            type_class = 1
        else:
            type_class = 0

    elif x < 0 and y < 0:
        if np.sqrt(x ** 2 + (y + 1) ** 2) <= 1 and np.sqrt((x + 1) ** 2 + y ** 2) <= 1:
            type_class = 1
        else:
            type_class = 0

    else:
        print("error!")
        exit(1)

    return type_class


def create_dataset(n_samples: int) -> (np.array, np.array, np.array, np.array):
    x_test = np.random.uniform(-1, 1, size=(n_samples, 2))
    y_test = np.array([get_class(point[0], point[1]) for point in x_test])
    x_train = np.random.uniform(-1, 1, size=(n_samples, 2))
    y_train = np.array([get_class(point[0], point[1]) for point in x_train])

    return x_train, x_test, y_train, y_test


def build_model_classifier(input_shape: int, n_nodes: int, output_size: int) -> keras.Sequential:
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Dense(n_nodes, activation=keras.activations.relu))
    model.add(keras.layers.Dense(n_nodes, activation=keras.activations.relu))
    model.add(keras.layers.Dense(n_nodes, activation=keras.activations.relu))
    model.add(keras.layers.Dense(n_nodes, activation=keras.activations.relu))
    model.add(keras.layers.Dense(output_size, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.binary_accuracy])
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


def evaluate_deep_learning(x_train: np.array, x_test: np.array, y_train: np.array, y_test: np.array, epochs: int):
    y_train_fit = keras.utils.to_categorical(y_train, 2)

    y_test_fit = keras.utils.to_categorical(y_test, 2)

    model = build_model_classifier(x_train[0].shape, 20, y_test_fit[0].size)

    callbacks = [
        keras.callbacks.TensorBoard(log_dir="log_dir", histogram_freq=1)
    ]

    history = model.fit(x_train, y_train_fit, epochs=epochs, batch_size=10,
                        validation_data=(x_test, y_test_fit), callbacks=callbacks).history

    plot_training("Training loss", "Validation loss", "Petals Classification Loss function", "Loss",
                  history["loss"], history["val_loss"], range(1, epochs + 1))

    plot_training("Training acc", "Validation acc", "Petals Classification Accuracy Function", "Accuracy",
                  history["binary_accuracy"], history["val_binary_accuracy"], range(1, epochs + 1))

    y_pred = [np.argmax(predict) for predict in model.predict(x_test)]

    result = metrics.accuracy_score(y_test, y_pred)
    matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(matrix)
    print(result)


def evaluate_no_linear_svm(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    svm = SVM.NuSVC(gamma="auto")
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    result = metrics.accuracy_score(y_test, y_pred)
    matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(result)
    print(matrix)
    # plot_svm_decision_function(svm, x_test, y_test, "SVM decision function")


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = create_dataset(3000)

    # plot_function(x_train[:, 0], x_train[:, 1], "teste", class_to_color(y_train))

    # evaluate_no_linear_svm(x_train, y_train, x_test, y_test)
    # evaluate_deep_learning(x_train, x_test, y_train, y_test, 300)
