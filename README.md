# neural-networks-project
This project contains the implementation, using the PyTorch framework, of SegNet, 
U-Net and Fully Convolutional Network (FCN) models for the automatic segmentation 
of objects belonging to the Cambridge-Driving Labeled Video Database (CamVid) 
dataset.

## Pipeline
### Data and preprocessing
The work was developed starting from the data contained within the [CamVid 
dataset](https://doi.org/10.1016/j.patrec.2008.04.005). This dataset is designed for 
semantic segmentation, providing video frames taken from the perspective of a moving car. 

Each pixel in the image is associated with a specific semantic class; the complete 
list of available classes (with the corresponding RGB code of the segmentation masks) 
is available in the _class_palette.csv_ file. There are also pixels that do not 
belong to any of these classes, and these were ignored in the calculation 
of the loss metrics.

To increase the diversity of the data passed to the model and improve its 
ability to generalize, images were randomly cropped and resized to a size 
$224\times224$, then possibly mirrored horizontally and then had pixel 
values normalized.

### The model
For this work, the [SegNet](https://doi.org/10.1109/TPAMI.2016.2644615), [U-Net](https://doi.org/10.1007/978-3-319-24574-4_28), 
and [FCN](https://doi.org/10.48550/arXiv.1411.4038) models were implemented. For all models, the developed 
implementations allow both training from scratch, initializing the encoder layers from scratch, 
and the use of layers from the [VGG16](https://doi.org/10.48550/arXiv.1409.1556) network trained on the ImageNet dataset.

In the case of the U-Net model, some modifications were made compared to the original implementation, 
as the adopted implementation does not use cropping or zero padding, thus preserving the spatial 
dimensions between input and output.

### Training and evaluation loop
The dataset is typically provided with a data split into training, validation, and 
test sets of $367\ (52\%)$, $101\ (15\%)$, and $233\ (33\%)$, respectively. This data 
split is the most commonly used in work using this dataset (such as 
[SegNet](https://doi.org/10.1109/TPAMI.2016.2644615)), and has therefore been retained here as well.

[Optuna](https://optuna.org/) is used for hyperparameters optimization (if specified using command line flags), 
otherwise the hyperparameters can be manually adjusted. 

In both cases, the train set is used to train the model and the validation set to evaluate its 
performances. Early stopping is used to stop the model training. Cross entropy is used as loss function.

If automatic hyperparameters optimization is performed, then the best model parameters are used 
to train a new model from scratch; if optimization was not selected, then the command-line provided 
arguments are used. Train and validation sets are used together to train the new model, which is 
then evaluated on the test set.
## Command line interface
```
python3 data_folder_root_filepath main.py [-h] [--image-size IMAGE_SIZE] [--model {unet, segnet, fcn}] 
[--lr LR] [--momentum MOMENTUM] [--n-trials N_TRIALS] [--batch-size BATCH_SIZE] [--epochs EPOCHS] 
[--patience PATIENCE] [--min-delta MIN_DELTA] [--vgg] [--enable-optimization] [--test]        
```

The first positional argument ```data_folder_root_filepath``` is the path to the project data directory, 
containing the dataset folders and the .csv file containing the classes labels.

```--image-size``` allows to specify the size (width and height) of the images after resize and crop. The 
default value is $224$.

```--model``` let the user specify the architecture to use, among SegNet, U-Net and FCN. U-Net is the default one.

```--vgg``` loads the VGG16 encoder layers and uses their weights (pretrained on ImageNet) to initialize the model's weights.

```--lr```, ```--momentum```, ```--batch-size``` and ```--epochs``` let the user specify the learning rate, 
momentum, batch size and maximum number of epochs for hyperparameter optimization and final retraining of 
the best model. 

```--enable-optimization``` enables the search for best hyperparameters, using ```--n-trials``` as the number of trials to perform. 

```--patience``` and ```--min-delta``` are the parameters to set up the Early Stopping. 

With ```--test``` is possible to specify whether to retrain the best model on the test and validation set, and evaluate it on the test.


