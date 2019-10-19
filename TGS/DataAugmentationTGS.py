import imgaug as ia
from imgaug import augmenters as iaa
import random
from random import randint
import gc

from tqdm import tqdm
import numpy as np

def affineDataAug():
    affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-20, 20),
                           translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode='symmetric'),
                ]),
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.02, 0.04))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.10))),
    ], random_order=True)

    #affine_seq_det = affine_seq.to_deterministic()
    return affine_seq



def augmenteTrainValid(x_train, y_train, iaseq, number_aug = 2000):

    b, img_size_target, img_size_target, d = x_train.shape

    _x = []
    _y = []
    for i in tqdm(range(0, number_aug)):
        index = random.randint(0,x_train.shape[0]-1)
        affine_seq_det = iaseq.to_deterministic()
        _x.append(affine_seq_det.augment_image(x_train[index]) )
        _y.append( affine_seq_det.augment_image(y_train[index]))

    x_train_aug = np.array(_x).reshape(-1, img_size_target, img_size_target, 1)
    y_train_aug = np.array(_y).reshape(-1, img_size_target, img_size_target, 1)

    del _x, _y
    gc.collect()

    return x_train_aug, y_train_aug

#x_train = np.append(x_train, [affine_seq_det.augment_image(x) for x in x_train], axis=0)
#y_train = np.append(y_train, [affine_seq_det.augment_image(x) for x in y_train], axis=0)