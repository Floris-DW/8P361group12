import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
import numpy as np
import scipy.signal


def MSE(OG, NW):
    """ just the MeanSquaredError """
    return np.square(OG - NW).mean()



def NCC_rgb(input_images, reconstructed_images):
    final_cc = []

    for i in range(32):
        cc_rgb = tf.zeros((1,1,1,1))
        for j in range(3):
            
            kernel = tf.reshape(reconstructed_images[i,:,:,j], [96,96,1,1])
            matrix = tf.reshape(input_images[i,:,:,j], [1,96,96,1])
            cc_c = tf.nn.conv2d(matrix,kernel ,1, padding = "VALID")
            cc_rgb = tf.concat([cc_rgb, cc_c],0)
            
        cc = tf.math.reduce_sum(cc_rgb) / (3*96*96) 
        cc =  1 - cc
        cc = tf.reshape(cc,[1])
        final_cc = tf.concat([final_cc, cc],0)
    return final_cc

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
    test_something = True
    if test_something:
        import autoencoder
        #%% loss function test
        test_gen_H, test_gen_D = autoencoder.ImageGeneratorsTest("../../Images/")
        images = test_gen_H.next()
        print(NCC_rgb(images,images))

    test_perceptual_loss = False
    if test_perceptual_loss:
        path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
        path_models = './models/'
        from autoencoder import ImageGeneratorsTrain, TrainModel, LoadModel
        model = LoadModel()
        num_epochs = 1
        train_gen, val_gen = ImageGeneratorsTrain(path_images)
        history = TrainModel(model, train_gen, val_gen, num_epochs,loss=perceptual_loss, save_model=False)

