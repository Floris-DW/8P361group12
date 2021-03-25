"""
source for formula's
https://stats.stackexchange.com/questions/11421/what-is-the-difference-between-empirical-variance-and-variance
"""
import os
import numpy as np
import PIL
import time
#%%
t1 = time.time()
t2 = t1

path_images = '../../Images/' # navigate to ~/source/Images from ~/source/Github/autoencoder.py
#os.listdir(path_models)

paths = {'Train':         '/'.join((path_images,'train+val/train','0')),
         'Valdation':     '/'.join((path_images,'train+val/train','0')),
         'Test Healthy':  '/'.join((path_images,'train+val/valid','0')),
         'Test Diseased': '/'.join((path_images,'train+val/valid','1'))}

print('[info] starting')
for key,value in paths.items():
    print(f'[info] working on {key}')
    L_images = os.listdir(value)
    tot = len(L_images)

    mean = 0 # is an numpy image array
    for i,p in enumerate(L_images):
        im = PIL.Image.open(value+'/'+p)
        im = np.asarray(im)/255
        mean += im
        print(f'\r[progres - mean] {i+1}/{tot}',end="")
    mean = mean/tot

    "Ei(Xi - mean)^2"
    Ex_m = 0 # is an numpy image array
    for i,p in enumerate(L_images):
        im = PIL.Image.open(value+'/'+p)
        im = np.asarray(im)/255
        Ex_m += (im-mean)**2
        print(f'\r[progres - Ei(xi-mean)^2] {i+1}/{tot}',end="")
    variance =       Ex_m/tot
    sample_variace = Ex_m/(tot-1)
    standard_error = (Ex_m/(tot*(tot-1)))**0.5
    print('\r[results]                            ')
    print(f'mean :            {np.mean(mean)          :.4f} | {np.mean(mean)*255          : 9.4f}')
    print(f'variance :        {np.mean(variance)      :.4f} | {np.mean(variance)*255      : 9.4f}')
    print(f'sample variance : {np.mean(sample_variace):.4f} | {np.mean(sample_variace)*255: 9.4f}')
    print(f'standard error :  {np.mean(standard_error):.4f} | {np.mean(standard_error)*255: 9.4f}')
    t3 = time.time()
    print(f'[info] This set took:   {(t3-t2)//3600:.0f}u {(t3-t2)%3600//60:.0f}m {(t3-t2)%3600%60:.0f}s')
    t2 = t3
t3 = time.time()
print(f'[info] Total time used: {(t3-t1)//3600:.0f}u {(t3-t1)%3600//60:.0f}m {(t3-t1)%3600%60:.0f}s')


