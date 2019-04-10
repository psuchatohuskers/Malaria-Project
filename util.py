import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt


def sigmoid(A):
	return 1/(1+np.exp(-A))

def sigmoid_cost(T, Y):
	return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def error_rate(targets,predictions):
	return np.mean(targets != predictions)

def read_image(path):
    img_file_list = glob.glob(path+"*.png")
    img_list = [cv2.imread(i) for i in img_file_list]
    return img_list

def resize_flattern(image_list,horizon_pix=64,vertical_pix=64):
    resize = []
    for img in image_list:
        reshape_pic = cv2.resize(img,(horizon_pix,vertical_pix))
        resize.append(reshape_pic.flatten())
    return np.array(resize)

def get_data(path1,path2,horizon=64,vertical=64):
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

def display_img(flat_img,label,horizon=64,vertical=64,col=3):
   	plt.imshow(flat_img.astype(int).reshape(horizon,vertical,col))
   	plt.title(label)
   	plt.show()
