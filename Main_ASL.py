import os
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from tensorflow.python.framework import ops
import cv2
import math

import time
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib

#%%
data = np.zeros([96,128,128,210])
for i in range(0,96):
    data[i,:,:,:]= np.load("/kaggle/input/asldata/ASL_data{}.npy".format(i))
   # print(i)
#%%
def prepareData(data):
    index, row, col,sli = data.shape
    
    temp = np.transpose(data, (1,2,0,3))   
    temp = np.reshape(temp,(row,col,-1))
    bigy = np.transpose(temp, (2,0,1))
    print(bigy.shape)
    
    # convert to k-space
    imgs, row, col = bigy.shape
    
    bigx = np.empty((imgs, row, col, 2))
    print('Preparing data')
    for i in range(imgs):
        #print(str(i+1)+ ' of ' +str(imgs-1))
        bigx[i, :, :, :] = create_x(np.squeeze(bigy[i,:,:]), normalize=False)
        
    # convert bigx from complex to abs values
    # bigx = np.abs(bigx)
    plt.imshow(bigy[100,:, :]),plt.xticks([]), plt.yticks([])
    return bigx,bigy


def create_x(y, normalize=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    x = to_freq_space(y)  # FFT: (128, 128, 2)
    x = np.expand_dims(x, axis=0)  # (1, 128, 128, 2)
    if normalize:
        x = x - np.mean(x)

    return x


def to_freq_space(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """

    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag

tic1 = time.time()
X_train, Y_train = prepareData(data)
toc1 = time.time()
print('Time to load and prepare data = ', (toc1 - tic1))
print('X_train.shape at input = ', X_train.shape)
print('Y_train.shape at input = ', Y_train.shape)

#%%
(m, n_H0, n_W0, _) = X_train.shape

tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_H0, n_W0, 2], name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, n_H0, n_W0], name='y')

x_temp = tf.keras.layers.Flatten()(x)  # size (n_im, n_H0 * n_W0 * 2)
n_out = np.int(x.shape[1] * x.shape[2])  # size (n_im, n_H0 * n_W0)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(x_temp.shape[1]))
#model.add(tf.keras.layers.Dense(n_out))
model.add(tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding='same'))
#model.add(tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.Conv2D(1, 7, strides=(1, 1), padding='same'))

#%%
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy)
model.fit(X_train, Y_train, batch_size=None, epochs=1, verbose=1)