The folder structure is as follows:

Report-models
 |- README.txt
 |- ModellenExperiment1
     | - [model description]
           |- XXX.json
           |- XXX.hdf5
           |- python_terminal_output [.txt / .docx] (optional)
 |- ModellenExperiment2
     | - [model description]
           |- XXX.json
           |- XXX.hdf5
           |- python_terminal_output [.txt / .docx] (optional)

The models are loaded using both the .json and .hdf5 file.
To load one of these models, provide the full path to the weights to the name variable.

The model description in the folder name works as follows,
for experiment 1, it is the loss used for training folowed by the size of the bottleneck. For experiment 2 it is "CNND" folowed by the dense layer based bottleneck size, and then the loss used for training.
