""" refrance source: https://github.com/seasonyc/densenet """
version = 'AE_v2' # naming sceme for models, prevent namingconflicts. AE = AutoEncoder

# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

import time
import numpy as np
import matplotlib.pyplot as plt


#%% just a timer utility
def _timer(func):
    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print (f'Runtime of "{func.__name__}" took {t2-t1:.2f} seconds')
        return f
    return inner

#%% all importable functions

def AutoEncoder(input_shape=(96,96,3), filters=[64, 32, 16], kernel_size=(3,3), pool_size=(2,2), dense_bn=None,
                activation='relu',padding='same', model_name=None):
    #encoder
    input_layer = Input(shape=input_shape)
    x = input_layer
    for i in filters:
        x = Conv2D(i, kernel_size, activation=activation, padding=padding)(x)
        x = MaxPool2D(pool_size=pool_size, padding=padding)(x)

    # dense layer bottlenek
    if dense_bn:
        s = x.shape[3] # save shape
        x = Dense(dense_bn, activation='relu')(x) # apply the bottlenek

        # restore the size
        x = Dense(s, activation='relu')(x)

    # decoder
    for i in filters[::-1]: #loop over the filter in reverse
        x = Conv2D(i, kernel_size, activation=activation, padding=padding)(x)
        x = UpSampling2D(pool_size)(x)
    x = Conv2D(input_shape[2], kernel_size, activation="sigmoid", padding=padding)(x) # output layer

    # assign the model an default name if None was given
    r = lambda x: str(x).replace(', ','.')[1:-1] # remove the (,),[,] and replace , with .
    model_name = model_name or f"{version}_F{ r(filters) }_K{ r(kernel_size) }_P{ r(pool_size) }_Dbn{dense_bn}"
    return Model(input_layer, x, name=model_name)


@_timer
def TrainModel(model, train, validation, num_epochs,
               loss='MeanSquaredError', optimizer='adam',
               save_model=True, verbose=1, save_dir='./models/'):
    if save_model:
        model_json = model.to_json() # serialize model to JSON
        with open(save_dir + model.name+'.json', 'w') as json_file:
            json_file.write(model_json)
        # prepare to save wheights
        date = time.strftime("%d-%m-%Y_%H-%M-%S")
        callbacks = ModelCheckpoint(save_dir + model.name+f'_W_E{num_epochs}_D{date}.hdf5', monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')
    else:
        callbacks=None

    model.compile(optimizer,loss)
    model_history = model.fit(train, epochs=num_epochs, batch_size=32, verbose=verbose,
                                validation_data=validation, callbacks=callbacks)
    return model_history

@_timer
def LoadModel(name='', path_models='./models/'):
    """
    load diffrent types of models.
    name: name of the model.
        - if '', selects the newest model
        - if no weights are given, picks latest available for the given name.

    **please use the weights (.hdf5) to refer to the model**
    """
    # if the given name does not contain the wheights or if no name given
    if not(name.endswith(".hdf5")):
        files = [path_models + x for x in os.listdir(path_models) if x.startswith(name) and x.endswith(".hdf5")]
        name = max(files , key = os.path.getctime).replace(path_models,'')

    i = name.upper().find("_W")
    model_path  = path_models + name[:i] + '.json'
    weight_path = path_models + name

    # load model and model weights
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)
    return model


def ImageGeneratorsTrain(base_dir, train_batch_size=32, val_batch_size=32, IMAGE_SIZE=96, split=0.3):
     """
     create twe genarators:
     train and validation.
     the train and validaion set are build from the split training set.
     """
     # dataset parameters
     train_path = base_dir + 'train+val/train'
     # we split the training set into training and validation.
     datagen_train = ImageDataGenerator(rescale=1./255, validation_split=split)
     # using the method from here might be better:
     # https://towardsdatascience.com/addressing-the-difference-between-keras-validation-split-and-sklearn-s-train-test-split-a3fb803b733
     train_gen = datagen_train.flow_from_directory(train_path, batch_size=train_batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             subset='training',   classes='0', class_mode='input')

     val_gen = datagen_train.flow_from_directory(train_path, batch_size=val_batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             subset='validation', classes='0', class_mode='input')
     return train_gen, val_gen


def ImageGeneratorsTest(base_dir, batch_size=32, IMAGE_SIZE=96):
     """
     create two genarators:
     test_healthy, test_diseased
     the test sets are the original validaion set.
     """
     # dataset parameters
     valid_path = base_dir + 'train+val/valid'
     #The test set is made from the validaion set
     datagen_test = ImageDataGenerator(rescale=1./255)
     # the test generators
     test_gen_H = datagen_test.flow_from_directory(valid_path, batch_size=batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             classes='0', class_mode=None)
     test_gen_D = datagen_test.flow_from_directory(valid_path, batch_size=batch_size, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             classes='1', class_mode=None)
     return test_gen_H, test_gen_D


def plot(model,gen):
    images = gen.next(); n=10
    decoded_imgs =  model.predict(images,batch_size=gen.batch_size)

    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


@_timer
def score(model, gen, loss, n = 16, verbose=False): # 16*32 = 512
    """ make a list of predictions based on a given loss function """
    bs = gen.batch_size
    l = np.zeros(n*bs)
    for i in range(n):
        O = gen.next()
        P = model.predict(gen,batch_size=bs) # give it the batch size to stop it from complaining
        l[i*bs:(i+1)*bs] = list(map(loss, O,P))
        if verbose: print(f'\rbatch {i}/{n}',end='')
    if verbose: print(f'\rbatch {n}/{n}')
    return l

def _TestScore(model,loss):
    test_gen_H, test_gen_D = ImageGeneratorsTest(path_images)
    #generate predictions for a 50/50 sample set.
    Hl = score(model, test_gen_H, loss, verbose=True,n=5)
    Dl = score(model, test_gen_D, loss, verbose=True,n=5)
    # the used loss function is NOT between 1 & 0 so normalize the data.
    tmp = np.max([Hl,Dl])
    Hl /= tmp
    Dl /= tmp

    labels = np.concatenate((np.zeros(Hl.size),np.ones(Dl.size)))
    predictions = np.concatenate((Hl,Dl))

    from sklearn.metrics import roc_curve, auc
    fpr1, tpr1, thresholds = roc_curve(labels, predictions)
    auc = auc(fpr1, tpr1)
    print(auc)
    #plotting ROC from random classifier
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr1, tpr1)
    plt.legend(["random","model"])
    #%%
    plt.figure()
    import scipy.stats
    color = {0:['green','springgreen','healthy'], 1:['red','tomato','diseased']}
    for T,data in enumerate([Hl,Dl]):
        bw =  None #0.1 # How smooth/rough the line can be, lower is rougher. None = default
        kde = scipy.stats.gaussian_kde(data,bw_method=bw)
        # plot (normalized) histogram of the data
        plt.hist(data, 50, density=1, facecolor=color[T][0], alpha=0.5);
        # plot density estimates
        t_range = np.linspace(0,1,200)
        plt.plot(t_range,kde(t_range),lw=2,
                 label=f'{color[T][0]} = {color[T][2]}',color=color[T][1])
        plt.xlim(0,1)
        plt.legend(loc='best')


if __name__ == '__main__':
    # Preparations
    path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
    path_models = './models/'

    if False:
        model = AutoEncoder()
        num_epochs = 4
        train_gen, val_gen = ImageGeneratorsTrain(path_images)
        history = TrainModel(model, train_gen, val_gen, num_epochs)
    else:
        model = LoadModel()
    #model.summary()

    import loss
    _TestScore(model, loss.MSE)

    #test_gen_H, test_gen_D = ImageGeneratorsTest(path_images)
    #plot(Model,test_gen_H)

