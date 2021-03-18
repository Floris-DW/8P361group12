import numpy as np
import scipy.signal
import modelmaker

def MSE_rgb(im1,im2):
    '''Input: two NxNx3 arrays
    Output: mean squared error value averaged for every color channel
    '''
    err_array = [0,0,0]
    for i in range(3):
        err = np.sum((im1[:,:,i] - im2[:,:,i]) ** 2)
        err = err / (im1[:,:,i].shape[0] * im1[:,:,i].shape[1])
        err_array[i] = err

    mse = np.mean(err_array)
    return mse



def NCC_rgb(im1,im2):
    '''Input: two NxNx3 arrays
    Output: mean of NCC values for all three color channels
    '''
    

    cc_array= [1,1,1]


    for i in range(im1.shape[2]):
        nim1 = im1[:,:,i] - im1[:,:,i].mean()
        nim2 = im2[:,:,i] - im2[:,:,i].mean()
        cc_array[i] = scipy.signal.correlate2d(nim1,nim2, mode = "valid")/(nim1.shape[0]*nim1.shape[1])
        
    cc = np.mean(cc_array)
    return cc

#%%
test_gen_H, test_gen_D = modelmaker.ImageGeneratorsTest("../../Images/")

images = test_gen_H.next()

print(NCC_rgb(images[0],images[0]))

