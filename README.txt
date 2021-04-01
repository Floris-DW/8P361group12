This zip file contains all code necessary to reproduce our results.

The same content can be found on our GitHub: https://github.com/tueimage/8p361-project-imaging

The folder structure is as follows:
8P361group12
   |- main.py		- the main python file, it can be used to reproduce our results.
   |- autoencoder.py	- a support python file with several functions used by main.py.
   | - loss.py		- a support python file with loss functions used by main.py.
   | - Image-stats.py	- a python script that was used to calculate the mean and variance of the datasets.
   | - models
        |- XXX.json
        |- XXX.hdf5		- a pretrained model.
        |-Report-models.zip	- a zip file containing all models used for the report.
        |- Report-models	- a folder containing all models used for the report.
              |-README.txt	- this readme explains how to find and import the pretrained models.
              |-ModellenExperiment1	- folder containing pretrained models
              |-ModellenExperiment2	- folder containing pretrained models

Please note that, by default, the Python scripts expect the images used for training and testing to be located in the following location:

Folder
   |-Images
   |-Github (the name of this folder does not matter)
        |- 8P361group12 (the name of this folder does not matter)
             |-main.py
             |-etc.
