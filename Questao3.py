import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches

#TODO conseguir recuperar os pontos dos desenhos

def plot_data():
    pyplot.xlim(-2, 2)
    pyplot.ylim(-2, 2)
    ax = pyplot.gca()
    r = 0.7
    class_1 = patches.Wedge((0, -1), r, 0, 180, fill=True, color="red")
    class_2 = patches.Wedge((1, 0), r, 90, 270, fill=True, color="black")
    class_3 = patches.Wedge((-1, 0), r, -90, 90, fill=True, color="green")
    class_4 = patches.Wedge((0, 1), r, -180, 0, fill=True, color="yellow")
    class_5 = pyplot.Rectangle((-1, -1), 2, 2, fill=True, color="blue")
    ax.add_patch(class_5)
    ax.add_patch(class_1)
    ax.add_patch(class_2)
    ax.add_patch(class_3)
    ax.add_patch(class_4)

    pyplot.show()

if __name__ == '__main__':
    plot_data()
