import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
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


vgg19_model = VGG19(include_top=False, weights="imagenet", input_shape=(96,96,3))
vgg19_model.trainable = False
def perceptual_loss(input_images, reconstructed_images):
    """
    Calculates the relative perceptual L1 loss between an original image and its reconstructed image.
    Designed to work with the existing code in classifier.py and the PCAM image generators.

    Expected input: numpy array with shape: (number of images (batch size),X,Y,Z)
    Output: Relative perceptual L1 loss (float)
    """
    # pre-calculated mean and std of the filter responses of the vgg19 model
    vgg19_mean = tf.reshape(tf.constant([0.485, 0.456, 0.406]),(1,3,1,1))
    vgg19_std = tf.reshape(tf.constant([0.229, 0.224, 0.225]),(1,3,1,1))

    features_1 = vgg19_model(input_images)
    features_2 = vgg19_model(reconstructed_images)

    f1_n = (features_1-vgg19_mean)/vgg19_std
    f2_n = (features_2-vgg19_mean)/vgg19_std

    num = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(f1_n - f2_n),axis=[1,2,3]))
    den = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(f2_n),axis=[1,2,3]))

    losses = num/den

    return losses

# put your functionality tests in here:
if __name__ == '__main__':
    test_something = False
    if test_something:
        import autoencoder
        #%% loss function test
        test_gen_H, test_gen_D = autoencoder.ImageGeneratorsTest("../../Images/")
        images = test_gen_H.next()
        print(NCC_rgb(images[0],images[0]))

    test_perceptual_loss = False
    if test_perceptual_loss:
        path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
        path_models = './models/'
        from autoencoder import ImageGeneratorsTrain, TrainModel, LoadModel
        model = LoadModel()
        num_epochs = 1
        train_gen, val_gen = ImageGeneratorsTrain(path_images)
        history = TrainModel(model, train_gen, val_gen, num_epochs,loss=perceptual_loss, save_model=False)

