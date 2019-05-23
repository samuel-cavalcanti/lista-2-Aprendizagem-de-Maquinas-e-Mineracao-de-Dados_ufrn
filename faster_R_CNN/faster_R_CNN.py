import pandas as pd

import keras
import cv2
import numpy as np

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')


def build_model(weights_path="None") -> keras.Sequential:
    model = keras.Sequential()

    # input layer
    model.add(keras.layers.InputLayer(input_shape=(224, 224, 3)))

    # block 1
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block1_conv1"))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block1_conv2"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="vgg_block1_pool"))

    # block 2
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block2_conv1"))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block2_conv2"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="vgg_block2_pool"))

    # block 3
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block3_conv1"))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block3_conv2"))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block3_conv3"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="vgg_block3_pool"))

    # block 4
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block4_conv1"))
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block4_conv2"))
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block4_conv3"))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="vgg_block4_pool"))

    # block 5
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block5_conv1"))
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block5_conv2"))
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name="vgg_block5_conv3"))

    # block 6

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation="relu", name="vgg_block6_fc1"))
    model.add(keras.layers.Dense(4096, activation="relu", name="vgg_block6_fc2"))
    model.add(keras.layers.Dense(1000, activation="softmax", name="output_NN"))

    model.compile(optimizer=keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  loss=keras.losses.categorical_crossentropy)

    if weights_path == "None":
        weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        model.load_weights(weights_path)

        model.summary()

    return model


def load_data() -> (np.array, np.array, np.array):
    car = cv2.imread("car.jpg")
    phone = cv2.imread("phone.jpg")
    dog = cv2.imread("dog.jpg")
    cat_1 = cv2.imread("cat.jpg")
    cat_2 = cv2.imread("cat_like_dog.jpg")
    plant = cv2.imread("plant.jpg")

    return [car, phone, dog, cat_1, cat_2, plant]


def show_images(name_window: str, images: list, labels: list) -> None:
    for i, image in enumerate(images):
        window = name_window + str(i)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        for j, text in enumerate(labels[i]):
            cv2.putText(image, text, (image.shape[1] // 2 - 100, 50 + 50 * j), 3, 1, (0, 255, 0), 2)
        cv2.imshow(window, image)
    cv2.waitKey(0)


def images_to_pred_images(images: list):
    return [cv2.resize(image, (224, 224)) for image in images]


def show_image(name_window: str, image: np.array, labels: list) -> None:
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    for j, text in enumerate(labels):
        cv2.putText(image, text, (image.shape[1] // 2 - 100, 50 + 50 * j), 3, 1, (0, 255, 0), 2)
    cv2.imshow(name_window, image)
    cv2.waitKey(30)


def test_online(model: keras.models.Model):
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()
        if ret_val:
            pred = model.predict(np.array([cv2.resize(img, (224, 224))]))
            pred = keras.applications.vgg16.decode_predictions(pred)
            labels = [label[1] for label in pred[0]]
            show_image("Webcam", img, labels)


def test_offline(model: keras.models.Model):
    images = load_data()

    pred_images = images_to_pred_images(images)

    pred = model.predict(np.array(pred_images))

    pred = keras.applications.vgg16.decode_predictions(pred)

    labels = list()

    for p in pred:
        l = list()
        for label in p:
            l.append(label[1])
        labels.append(l)

    show_images("image", images, labels)


if __name__ == '__main__':
    model = keras.applications.VGG16()

    # test_offline(model)

    test_online(model)

