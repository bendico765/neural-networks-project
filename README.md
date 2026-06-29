# neural-networks-project

This project contains the implementation of an automatic segmentation algorithm for the segmentation of masses in mammography. 
## Pipeline
### Preprocessing
This work has been developed using data from the [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) dataset. Each mammography is first subject to the application, in order, of **median filtering**, **CLAHE** and **Unsharp masking**, and then both the image and the ground truth mask are cropped around the lesion. Finally, the image and the ground truth mask are resized to $512\times512$ and exported as Pytorch tensors.
### The model
### Training and evaluation loop
Data is divided into training, validation and test set using the 70-20-10 split ratio.

[Optuna](https://optuna.org/) is used for hyperparameters optimization (if specified using command line flags), otherwise the hyperparameters can be manually adjusted. 

In both cases, the train set is used to train the model and the validation set to evaluate its performances. Early stopping is used to stop the model training. Dice loss is used as loss function.

If automatic hyperparameters optimization is performed, then the best model parameters are used to train a new model from scratch; if optimization was not selected, then the command-line provided arguments are used. Train and validation sets are used together to train the new model, which is then evaluated on the test set.
## Command line interface
```
python3 data_folder_root_filepath main.py [-h] [--lr LR] [--n-trials N_TRIALS] [--batch-size BATCH_SIZE]
[--epochs EPOCHS] [--loss {dice,jaccard,bce}] [--patience PATIENCE] [--min-delta MIN_DELTA] [--random-state RANDOM_STATE]
[--enable-optimization] [--test]        
```

The first positional argument ```data_folder_root_filepath``` is the path to the project data directory, containing the dataset tensors and the .csv file containing the metadata of the lesions. 

```--lr```, ```--batch-size``` and ```--epochs``` let the user specify the learning rate, the batch size and the maximum number of epochs for hyperparameter optimization and final retraining of the best model. 

```loss``` allows to select the loss function to train the models, between Dice, Jaccard and BinaryCrossEntropy.

```--enable-optimization``` enables the search for best hyperparameters, using ```--n-trials``` as the number of trials to perform. 

```--patience``` and ```--min-delta``` are the parameters to set up the Early Stopping. 

With ```--test``` is possible to specify whether or not to retrain the best model on the test and validation set, and evaluate it on the test.

Finally, ```--random-state``` is the random-state to be used for the train-validation-test split.
