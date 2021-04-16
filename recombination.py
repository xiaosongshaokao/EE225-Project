import cv2
import numpy as np

# if don't have the folder, create it first
# import os
# os.mkdir('recombination')

for i in range(1, 101):
    img_path = 'outputs/' + str(i) + '.jpg'
    image = cv2.imread(img_path)
    for j in [0, 1, 2, 3, 649, 650, 651, 652, 653, 654, 655]:
        for k in range(0, 875):
            image[j][k] = 0
    new_image = np.ndarray((645, 875, 3), dtype=np.uint8)
    for j in range(0, 325):
        for k in range(0, 875):
            new_image[j][k][0] = image[j + 324][k][0]
            new_image[j][k][1] = image[j + 324][k][1]
            new_image[j][k][2] = image[j + 324][k][2]
    for j in range(325, 645):
        for k in range(0, 875):
            new_image[j][k][0] = image[j - 321][k][0]
            new_image[j][k][1] = image[j - 321][k][1]
            new_image[j][k][2] = image[j - 321][k][2]
    for j in [0, 1, 2, 650, 651, 652, 653, 654, 655]:
        for k in range(0, 875):
            image[j][k] = 0
    # new_image = np.ndarray((646, 875, 3), dtype=np.uint8)
    # print(image.shape)
    # for j in range(0, 325):
    #     for k in range(0, 875):
    #         new_image[j][k][0] = image[j + 324][k][0]
    #         new_image[j][k][1] = image[j + 324][k][1]
    #         new_image[j][k][2] = image[j + 324][k][2]
    # for j in range(325, 646):
    #     for k in range(0, 875):
    #         new_image[j][k][0] = image[j - 322][k][0]
    #         new_image[j][k][1] = image[j - 322][k][1]
    #         new_image[j][k][2] = image[j - 322][k][2]
    save_path = 'recombination/' + str(i) + '.jpg'
    cv2.imwrite(save_path, new_image)
