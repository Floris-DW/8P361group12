# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:37:29 2021

@author: 20183193
"""
"NOTE: curently we do not have a proper test set, so a subset of the validation set is used"

#%% imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import model_from_json
import modelmaker
import losses

#%% define paths
path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
path_models = './models/'

#model_path = path_models + r'09-03-2021_15-35-37_model_autoencoder_Ldim-2'

use_latest_model = True
if use_latest_model:
    files = [path_models + x for x in os.listdir(path_models) if x.endswith(".json")]
    model_path = max(files , key = os.path.getctime)[:-5]  # the [:-5] cuts off the .json


#%% define helper function.
# this helper function is now defined in both files.
# it may be beneficial to swich to a main() and import strategy later.
IMAGE_SIZE = 96
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, classes =['0'], class_mode='input'):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train+val', 'train')
     valid_path = os.path.join(base_dir, 'train+val', 'valid')

     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             classes=classes,
                                             class_mode=class_mode)

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             classes=classes,
                                             class_mode=class_mode)

     return train_gen, val_gen
#%% define mean squared error function for rgb images

def MSE_rgb(im1,im2):
    '''Input: two NxNx3 arrays
    Output: mean squared error value
    '''
    err_array = [0,0,0]
    for i in range(3):
        err = np.sum((im1[:,:,i] - im2[:,:,i]) ** 2)
        err = err / im1[:,:,i].shape[0] * im1[:,:,i].shape[1]
        err_array[i] = err

    mse = sum(err_array) / 3
    return mse

#%% load model and model weights
model = modelmaker.LoadModel()

#%% plot test:
n = 10

_, image_test = get_pcam_generators(path_images, train_batch_size=n, classes=['0'], class_mode=None)
images = image_test.next()
images = np.clip(images, 0, 1)

decoded_imgs =  np.clip(model.predict(images),0,1)
decoded_imgs =  np.clip(decoded_imgs, 0, 1)

plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(images[i])
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i])
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#%% Registration error healthy images

# real_img = images[0] - np.mean(images[0])
# recreated_img = decoded_imgs[0] - np.mean(decoded_imgs[0])

# u = real_img.reshape((real_img.shape[0]*real_img.shape[1],3))
# v = recreated_img.reshape((recreated_img.shape[0]*recreated_img.shape[1],3))

# u = u - u.mean(keepdims=True)
# v = v - v.mean(keepdims=True)
    
# CC=(np.transpose(u).dot(v))/(((np.transpose(u).dot(u))**0.5)*((np.transpose(v).dot(v))**0.5))
#print(np.mean(CC))
cc = losses.NCC_rgb(images[0], images[0])
print(cc)


#%%
mse_list_0 = []
cycles = 3
#loops over cycles x batch_size class 0 images and calculates mean squared error
for j in range(cycles):
    for i in range(images.shape[0]):
                
        images = np.clip(images, 0, 1)
        decoded_imgs =  np.clip(model.predict(images),0,1)
        decoded_imgs =  np.clip(decoded_imgs, 0, 1)

        real_img = images[i]
        recreated_img = decoded_imgs[i]
        images = image_test.next()
        mse_list_0.append(MSE_rgb(real_img, recreated_img))



#%%initiate unhealthy image generator
_,imggen_class1 = get_pcam_generators(path_images, classes=['1'], class_mode=None)
images_class1 = image_test.next()
images_class1 = np.clip(images, 0, 1)
#%% Registration error unhealthy images

mse_list_1 = []
for j in range(cycles):
    for i in range(images_class1.shape[0]):
                
        images_class1 = np.clip(images, 0, 1)
        decoded_imgs =  np.clip(model.predict(images_class1),0,1)
        decoded_imgs =  np.clip(decoded_imgs, 0, 1)
        real_img = images_class1[i]
        recreated_img = decoded_imgs[i]
        images = image_test.next()
        
        mse_list_1.append(MSE_rgb(real_img, recreated_img))


#%%
print("MSE healthy images:",np.mean(mse_list_0))
print('MSE unhealthy images', np.mean(mse_list_1))
