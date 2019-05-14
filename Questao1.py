from sklearn.linear_model.perceptron import Perceptron
from sklearn.svm import SVC as SVM
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_model(data: np.array, model) -> None:
    pyplot.figure("SVM")
    z = lambda _x, _y: (-model.intercept_[0] - model.coef_[0][0] * _x - model.coef_[0][1] * _y) / model.coef_[0][2]
    xx = np.linspace(0, 1, 3)

    x, y = np.meshgrid(xx, xx)
    ax = pyplot.axes(projection='3d')
    ax.scatter(data[0:4, 0], data[0:4, 1], data[0:4, 2], color="green", s=100)
    ax.scatter(data[4:, 0], data[4:, 1], data[4:, 2], color="red", s=100)
    ax.plot_surface(x, y, z(x, y), color="blue")

    pyplot.title("SVM plot")
    # pyplot.savefig("graficos/SVM", format='png', dpi=300)
    pyplot.show()


def test_svm(data: list, label: list) -> None:
    svm = SVM(gamma="scale", kernel="linear")
    svm.fit(data, label)
    plot_model(np.array(data), svm)


def test_perceptron(x: list, y: list, learning_rate: float, max_iter: int) -> None:
    perceptron = Perceptron(max_iter=max_iter, alpha=learning_rate)
    perceptron.fit(x, y)
    plot_model(np.array(x), perceptron)


def main():
    data = [
        # class C1
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        # class C2
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 1]
    ]
    label = [
        # class C1
        0, 0, 0, 0,
        # class C2
        1, 1, 1, 1
    ]
    test_perceptron(data, label, 0.01, 100)
    test_svm(data, label)

if __name__ == "__main__":
    main()

    pass
