# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 13:11:26 2017

@author: msamouiller
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_img_3planes(img,name='RGB'):
    fig = plt.figure()
    a=fig.add_subplot(2,3,1)
    plt.imshow(img)
    a.set_title(name)
      
    a=fig.add_subplot(2,3,4)
    plt.imshow(img[:,:,0], cmap='gray')
    a.set_title(name[0])
      
    a=fig.add_subplot(2,3,5)
    plt.imshow(img[:,:,1], cmap='gray')
    a.set_title(name[1])  
      
    a=fig.add_subplot(2,3,6)
    plt.imshow(img[:,:,2], cmap='gray')   
    a.set_title(name[2]) 

"""'%01d.tif' % 1"""

path = 'C:/Users/msamouiller/Documents/_Cours_Articles_Books_Exos/Python/Kaggle/SeaLion/TrainSmall/Train/'
str_img = '1_1.jpg'

img = cv2.imread(path +str_img, 1)
img_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure()
plt.imshow(img_opencv)
fig.text(0.5, 0.05,'"MSA" Image Sea Lion',ha='center')
        

        
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


display_img_3planes(HSV, 'HSV')
"""
s = hsv[:,:,1]
fig = plt.figure()
plt.imshow(s)

ret1,th1 = cv2.threshold(s,70,255,cv2.THRESH_BINARY)
fig = plt.figure()
plt.imshow(th1)"""
