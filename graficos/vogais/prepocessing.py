import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

'''
A = 0 I = 2
E = 1 O = 3
U = 4
'''


def type_class(char: str) -> int:
    if char == "A":
        return 0
    if char == "E":
        return 1
    if char == "I":
        return 2
    if char == "O":
        return 3
    # if char == "U":
    return 4


def show_image(name_window: str, image: np.array) -> None:
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_window, image)
    cv2.waitKey(0)


def image_to_binary(image: np.array) -> np.array:
    resized_image = cv2.resize(image, (20, 20))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                      cv2.THRESH_BINARY, 3, 1)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def pre_processing(dir_name: str) -> np.array:
    images = list()
    classes = list()
    for root, dirs, files in os.walk(dir_name):
        if files:
            for image_file in files:
                image_path = os.path.join(root, image_file)

                image = cv2.imread(image_path)
                binary_image = image_to_binary(image)

                images.append(binary_image)
                classes.append(type_class(root[-1]))

    return np.array(images), np.array(classes)


def saving_dataset(x: np.array, y: np.array, filename: str) -> None:
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    np.savez(filename, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    good_vowels_image, good_vowels = pre_processing("GoodImg")
    bad_vowels_image, bad_vowels = pre_processing("BadImag")

    all_vowels_images = np.concatenate((good_vowels_image, bad_vowels_image), axis=0)
    all_vowels = np.concatenate((good_vowels, bad_vowels), axis=0)

    saving_dataset(good_vowels_image, good_vowels, "good_vowels")
    saving_dataset(bad_vowels_image, bad_vowels, "bad_vowels")
    saving_dataset(all_vowels_images, all_vowels, "all_vowels")
