import numpy as np
import matplotlib.pyplot as pyplot
import keras
from sklearn import svm as SVM
from sklearn import metrics


def plot_function(x: np.array, y: np.array, fig_name: str, y_true=None) -> None:
    pyplot.figure(fig_name)
    pyplot.title(fig_name + " function")

    pyplot.plot(x, y, "o", label="Y_pred")
    if y_true is not None:
        pyplot.plot(x, y_true, "or", label="Y_true", markersize=1.5)
        pyplot.legend()

    pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


spiral_0 = lambda theta: (theta / 4 * np.cos(theta), theta / 4 * np.sin(theta))

spiral_1 = lambda theta: ((theta / 4 + 0.8) * np.cos(theta), (theta / 4 + 0.8) * np.sin(theta))


def generate_dateset(min_value: int, max_value: int, n_samples: int) -> (np.array, np.array):
    theta = np.random.uniform(min_value, max_value, n_samples)

    spiral_0_data = np.array(list(spiral_0(theta))).T
    spiral_1_data = np.array(list(spiral_1(theta))).T

    return np.concatenate((spiral_0_data, spiral_1_data), axis=0), \
           np.concatenate((np.zeros(n_samples), np.ones(n_samples)))


def generate_random_dataset(min_value: int, max_value: int, n_samples: int) -> (np.array, np.array):
    theta = np.random.uniform(min_value, max_value, n_samples)

    spiral_0_data = np.array(list(spiral_0(theta))).T
    spiral_1_data = np.array(list(spiral_1(theta))).T

    return np.concatenate((spiral_0_data, spiral_1_data), axis=0), \
           np.concatenate((np.zeros(n_samples), np.ones(n_samples)))


def build_model(input_shape: tuple, output_size: int) -> keras.Sequential:
    model = keras.Sequential()
    nodes = 20
    model.add(keras.layers.Dense(nodes, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(nodes, activation="relu"))
    model.add(keras.layers.Dense(nodes, activation="relu"))
    model.add(keras.layers.Dense(nodes, activation="relu"))
    model.add(keras.layers.Dense(output_size, activation=keras.activations.sigmoid))
    model.compile(optimizer="rmsprop", loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.binary_accuracy])
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


def evaluate_deep_learning(n_epochs: int, x_train: np.array, y_train: np.array, x_test: np.array,
                           y_test: np.array) -> None:
    model = build_model(tuple([x_train.shape[-1]]), 1)
    history = model.fit(x_train, y_train, batch_size=20, epochs=n_epochs, validation_data=(x_test, y_test)).history

    acc = history["binary_accuracy"]
    val_acc = history["val_binary_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    plot_training("Training loss", "Validation loss", "spiral Function loss", "Loss", loss, val_loss,
                  range(1, n_epochs + 1))
    plot_training("Training acc", "Validation acc", "spiral Function accuracy", "Accuracy", acc, val_acc,
                  range(1, n_epochs + 1))

    model.save("questao5_deeplearning")


def plot_svm_decision_function(svm: SVM.NuSVR, x_test: np.array, y_test: np.array, fig_name: str) -> None:
    interval = [-5.5, 6]
    xx, yy = np.meshgrid(np.linspace(interval[0], interval[1], 500),
                         np.linspace(interval[0], interval[1], 500))
    # plot the decision function for each datapoint on the grid
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pyplot.figure(fig_name)
    pyplot.title(fig_name)

    pyplot.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                  origin='lower', cmap=pyplot.cm.PuOr_r)
    pyplot.contour(xx, yy, Z, levels=[0], linewidths=2,
                   linestyles='dashed')

    pyplot.scatter(x_test[:, 0], x_test[:, 1], s=30, c=y_test, cmap=pyplot.cm.Paired,
                   edgecolors='k')
    pyplot.xticks(())
    pyplot.yticks(())
    pyplot.axis([interval[0], interval[1], interval[0], interval[1]])
    pyplot.savefig("graficos/" + fig_name, format="png")
    pyplot.show()


def evaluate_no_linear_svm(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    svm = SVM.NuSVC(gamma=5)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    result = metrics.accuracy_score(y_test, y_pred)
    matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(result)
    print(matrix)
    # plot_svm_decision_function(svm, x_test, y_test, "SVM decision function")


if __name__ == '__main__':
    n_epochs = 200
    x_train, y_train = generate_dateset(0, 20, 1000)
    x_test, y_test = generate_dateset(0, 20, 5000)
    # evaluate_deep_learning(n_epochs, x_train, y_train, x_test, y_test)
    evaluate_no_linear_svm(x_train, y_train, x_test, y_test)
