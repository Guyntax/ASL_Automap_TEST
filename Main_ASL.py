#import os
import numpy as np
import tensorflow as tf

import time
from matplotlib import pyplot as plt
from random import randrange


#%%
# path = r"C:/Users/Admin/Documents/Python/UPIR/npy/"
path = r"/home/didier/Documents/ASL_data/"

data = np.zeros([96,128,128,210])
for i in range(0,96):
    data[i,:,:,:]= np.load(path+"ASL_data{}.npy".format(i))
   # print(i)
#%%
def prepareData(data):
    index, row, col,sli = data.shape
    
    temp = np.transpose(data, (1,2,0,3))    
    temp = np.reshape(temp,(row,col,-1))
    bigy = np.transpose(temp, (2,0,1))
    #print(bigy.shape)
    
    # convert to k-space
    imgs, row, col = bigy.shape
    
    bigx = np.empty((imgs, row, col, 2))
    print('Preparing data')
    for i in range(imgs):
        print(str(i+1)+ ' of ' +str(imgs))
        bigx[i, :, :, :] = create_x(np.squeeze(bigy[i,:,:]), normalize=False)
        
    # convert bigx from complex to abs values
    # bigx = np.abs(bigx)
    #plt.imshow(bigy[100,:, :]),plt.xticks([]), plt.yticks([])
    return bigx


def create_x(y, normalize=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    x = to_freq_domain(y)  # FFT: (128, 128, 2)
    x = np.expand_dims(x, axis=0)  # (1, 128, 128, 2)
    if normalize:
        x = x - np.mean(x)

    return x


def to_freq_domain(img):
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


def undersample(singleSlice, percentage):
    undersampledSlice = np.zeros(np.shape(singleSlice))
    dim = singleSlice.shape[0]
    for i in range(0,dim):
        for j in range(0,dim):
            undersampledSlice[i,j] = singleSlice[i,j]
    #print(round(dim*percentage)**2)
    
    indexList = list(np.arange(0,(dim**2)))
    for i in range(0,round((dim**2)*percentage)):
        randNUM = randrange(len(indexList))
        index = indexList[randNUM]
        indexList.pop(randNUM)
        
        undersampledSlice[int(np.floor(index/dim)), int(index - dim*np.floor(index/dim))] = 0
            
    return undersampledSlice

def createTrainingData(targetData,percentage):
    trainingData = np.zeros(np.shape(targetData))
    for i in range(0,np.shape(targetData)[0]):
        trainingData[i,:,:,0] = undersample(targetData[i,:,:,0],percentage)
        trainingData[i,:,:,1] = undersample(targetData[i,:,:,1],percentage)
    return trainingData

def to_space_domain(img):
    complexImg = np.zeros(np.shape(img[:,:,:,0]),dtype=np.complex)
    for i in range(0,np.shape(img)[0]):
        for j in range(0,np.shape(img)[1]):
            for k in range(0,np.shape(img)[2]):
                complexImg[i,j,k] = np.complex(img[i,j,k,0],img[i,j,k,1])
    return np.abs(np.fft.ifft2(complexImg))
 
   
tic1 = time.time()

percentage = 0.2
targetData = prepareData(data)
del data
trainingData = createTrainingData(targetData,percentage)

# targetData = [None]*np.shape(targetData0)[0]
# trainingData = [None]*np.shape(targetData0)[0]
# for i in range(0,np.shape(targetData0)[0]):
#     targetData[i] = targetData0[i,:,:,:]
#     trainingData[i] = trainingData0[i,:,:,:]
    


toc1 = time.time()
print('Time to load and prepare data = ', (toc1 - tic1))
#print('trainingData.shape at input = ', targetData.shape)
#%%



 # percentage = 0.2
# dude = undersample(Y_train[45,:,:],percentage)
# plt.imshow(Y_train[45,:,:])
# plt.show()
# plt.imshow(dude)
# plt.show()   
    
#recontructedIMG = to_space_domain(trainingData)
# plt.imshow(originalImages[115,:,:]);plt.show()
#plt.imshow(recontructedIMG[115,:,:]);plt.show()


#%%
tf.debugging.set_log_device_placement(True)
#with tf.device('/GPU:0'):

(m, n_H0, n_W0, _) = trainingData.shape

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (32768,)))
# # model.add(tf.keras.layers.Dense(np.int(n_H0*n_W0*2)))
# model.add(tf.keras.layers.Dense(np.int(n_H0 * n_W0)))
# model.add(tf.keras.layers.Reshape((n_H0,n_W0,1)))
# 
model.add(tf.keras.layers.Conv2D(16, 5, strides=(1, 1), padding='same',input_shape=(128,128,2)))

model.add(tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.Conv2D(64, 5, strides=(1, 1), padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(2, 7, strides=(1, 1), padding='same'))



model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
model.summary()

model.fit(trainingData, targetData, batch_size=100, epochs=5, verbose=1)
img = model.predict(tf.reshape(trainingData[134,:,:,:],(1,128,128,2)))
#
plt.imshow(to_space_domain(img)[0,:,:])
plt.savefig("output.png")

print('FIN')