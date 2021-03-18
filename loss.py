import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
import scipy.signal

def MSE_rgb(im1,im2):
    '''Input: two NxNx3 arrays
    Output: mean squared error value averaged for every color channel
    '''
    err_array = [0,0,0]
    for i in range(3):
        err = np.sum((im1[:,:,i] - im2[:,:,i]) ** 2)
        err = err / (im1[:,:,i].shape[0] * im1[:,:,i].shape[1])
        err_array[i] = err
    return np.mean(err_array)


def MSE(OG, NW):
    """ just the MeanSquaredError """
    return np.square(OG - NW).mean()


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


def perceptual_loss(input_images, reconstructed_images):
    """
    Calculates the relative perceptual L1 loss between an original image and its reconstructed image.
    Designed to work with the existing code in classifier.py and the PCAM image generators.

    Expected input: numpy array with shape: (number of images (batch size),X,Y,Z)
    Output: Relative perceptual L1 loss (float)
    """
    batch_size = input_images.shape[0]
    image_shape = input_images.shape[1:] #(96,96,3)

    # retrieve vgg19 model
    vgg19_model = VGG19(include_top=False, weights="imagenet", input_shape=image_shape)
    vgg19_model.trainable = False

    # pre-calculated mean and std of the filter responses of the vgg19 model
    vgg19_mean = np.array([0.485, 0.456, 0.406])
    vgg19_std = np.array([0.229, 0.224, 0.225])

    features_1 = vgg19_model.predict(input_images)
    features_2 = vgg19_model.predict(reconstructed_images)

    features_shape = features_1.shape[1:] #(3,3,512)

    # initialize losses matrix
    losses = np.empty((batch_size,features_shape[0],features_shape[1]))

    for i in range(batch_size):
        f1=features_1[i]
        f2=features_2[i]

        # initialize normalized feature matrices
        f1_n = np.empty(features_shape)
        f2_n = np.empty(features_shape)

        # normalization
        for j in range(features_shape[0]):
            f1_n[j] = (f1[j] - vgg19_mean[j])/vgg19_std[j]
            f2_n[j] = (f2[j] - vgg19_mean[j])/vgg19_std[j]

        # calculate perceptual loss of current image
        numerator = np.sqrt(((f1_n - f2_n)**2).sum(-1))
        denominator = np.sqrt((f1_n**2).sum(-1))
        curr_loss = numerator/denominator
        losses[i] = curr_loss
    # calculate end-result of the perceptual losses (this is the L1 loss (I think))
    return losses.sum(2).sum(1).sum(0)/losses.size

# put your functionality tests in here:
if __name__ == '__main__':
    import autoencoder
    #%% loss function test
    test_gen_H, test_gen_D = autoencoder.ImageGeneratorsTest("../../Images/")
    images = test_gen_H.next()
    print(NCC_rgb(images[0],images[0]))

