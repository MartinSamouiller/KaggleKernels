# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:14:06 2017

@author: msamouiller
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


"""'%01d.tif' % 1"""

path = 'C:/Users/msamouiller/Documents/_Cours_Articles_Books_Exos/Python/Kaggle/SeaLion/TrainSmall/Train/'
str_img = '7.jpg'

image_array = []
image_name = []

img = cv2.imread(path +str_img, 1)

"""OpenCV represents RGB images as multi-dimensional NumPy arrays…but in reverse order!
This means that images are actually represented in BGR order rather than RGB!
All we need to do is convert the image from BGR to RGB:"""
img_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
bgr2xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

"""
image_array.append(img_opencv) 
image_name.append('RGB')
image_array.append(lab)
image_name.append('LAB')
image_array.append(hsv)
image_name.append('HSV')
image_array.append(hls)
image_name.append('HLS')
image_array.append(ycrcb)
image_name.append('YCrCb')
image_array.append(bgr2xyz)
image_name.append('XYZ')"""

"""
i = 0
for img in image_array:
  fig = plt.figure()
  
  a=fig.add_subplot(2,3,1)
  imgplot = plt.imshow(img)
  a.set_title(image_name[i])
  
  a=fig.add_subplot(2,3,4)
  imgplot = plt.imshow(img[:,:,0], cmap='gray')
  a.set_title(image_name[i][0])
  
  a=fig.add_subplot(2,3,5)
  imgplot = plt.imshow(img[:,:,1], cmap='gray')
  a.set_title(image_name[i][1])  
  
  a=fig.add_subplot(2,3,6)
  imgplot = plt.imshow(img[:,:,2], cmap='gray')
  a.set_title(image_name[i][2])    
  
  i += 1
"""
#HSV : H permet de segmenter herbe / eau / rocher
#on va estimer la probabilite que les sealions soit sur les rocher/eau/herbe

HSV = hsv
H = hsv[:,:,0]
#fig = plt.figure()
#plt.hist(H.ravel(),256)

ret1,th1 = cv2.threshold(HSV,70,255,cv2.THRESH_BINARY)
#fig = plt.figure()
#plt.imshow(th1)

#K-means sur l'image HSV 

Z = HSV.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((HSV.shape))
#fig = plt.figure()
#plt.imshow(res2)

#Kmean sur le plan HSV et retour en RGB pour avoir les couleurs moyennes des 3 plans 
RGBKmeans = cv2.cvtColor(res2, cv2.COLOR_HSV2RGB)
#fig = plt.figure()
#plt.imshow(RGBKmeans)

gray_image = cv2.cvtColor(RGBKmeans, cv2.COLOR_BGR2GRAY)
kernel = np.ones((9,9),np.uint8)
erosion = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
#fig = plt.figure()
#plt.imshow(erosion, cmap='gray')
#fig = plt.figure()
#plt.imshow(gray_image, cmap='gray')

#Supprime le bruit dans la zone
RGBKmeansBlur = cv2.medianBlur(RGBKmeans, 15)
fig = plt.figure()
plt.imshow(RGBKmeansBlur)




"""b = img[:,:,0]
 >>> b,g,r = cv2.split(img)
 >>> img = cv2.merge((b,g,r))"""