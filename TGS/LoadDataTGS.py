import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tqdm import tqdm


import os

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    #return img[:img_size_ori, :img_size_ori]

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i


def loadDataset():
    train_df = pd.read_csv("train.csv", index_col="id", usecols=[0])
    depths_df = pd.read_csv("depths.csv", index_col="id")
    train_df = train_df.join(depths_df)
    
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    
    train_df["images"] = [np.array(load_img("train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["masks"] = [np.array(load_img("train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    train_df["depth_class"] =  pd.cut(train_df.z, bins=5, labels=False)
    
    test_df["depth_class"] =  pd.cut(test_df.z, bins=5, labels=False)

    return train_df, test_df


def split_train_test_slat_coverage(train_df):
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.coverage_class, random_state=1337)

    return x_train, x_valid, y_train, y_valid

def split_train_test_depth(train_df):
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
        train_df.coverage.values,
        train_df.z.values,
        test_size=0.2, stratify=train_df.depth_class, random_state=1337)

    return x_train, x_valid, y_train, y_valid


"""
def kfold_depth():
    n_fold = 5
    depths = pd.read_csv(os.path.join(DATA_ROOT, 'depths.csv'))
    depths.sort_values('z', inplace=True)
    depths.drop('z', axis=1, inplace=True)
    depths['fold'] = (list(range(n_fold))*depths.shape[0])[:depths.shape[0]]
    print(depths.head())

"""