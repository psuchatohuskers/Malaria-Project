import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

PATH_INFECT = "cell_images/Parasitized/"
PATH_UNINFECT = "cell_images/Uninfected/"

# Helper functions
def read_image(path):
    img_file_list = glob.glob(path+"*.png")
    img_list = [cv2.imread(i) for i in img_file_list]
    return img_list

def display_img(image_list):
    for img in image_list:
        plt.imshow(img)
        plt.show()

def resize_flattern(image_list,horizon_pix=64,vertical_pix=64):
    resize = []
    for img in image_list:
        reshape_pic = cv2.resize(img,(horizon_pix,vertical_pix))
        resize.append(reshape_pic.flatten())
    return np.array(resize)

def get_train_test(path1,path2,horizon=64,vertical=64):
    infected = read_image(path1)
    uninfected = read_image(path2)
    resize1 = resize_flattern(infected,horizon,vertical)
    resize0 = resize_flattern(uninfected,horizon,vertical)
    N1 = resize1.shape[0]
    N0 = resize0.shape[0]
    infected_label = np.hstack((np.ones((N1,1)),resize1))
    uninfected_label = np.hstack((np.zeros((N0,1)),resize0))
    clean_data = np.vstack([infected_label,uninfected_label])
    return clean_data


infected = read_image(PATH_INFECT)
resize = resize_flattern(infected)
N = resize.shape[0]
D = resize.shape[1]
plt.imshow(resize[0,:].reshape(64,64,3))
print(resize[0,:])
plt.show()
infected_lable = np.hstack((np.ones((N,1)),resize))
print(infected_lable[0,:])
# data = get_train_test(PATH_INFECT,PATH_UNINFECT)
# plt.hist(data[:,0])
# plt.show()


