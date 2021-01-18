# Python command-line Bengali Classificator

Module for training model on [competition data](https://www.kaggle.com/c/bengaliai-cv19) and predicting on a single image or batch of images using trained model.
Uses model from this [Notebook](https://www.kaggle.com/amanmishra4yearbtech/bengali-classification-quick-implementation).

## Getting Started

To download project:
```
git clone https://github.com/KopinHruk/cuddly-crankshaft.git 
```



## Usage without installation

### Prediction:
```
python  predict.py  images_path
```

#### Command line arguments:
`images_path`  -  Path to image or directory with images to predict on.

`--weights_path` - Path for loading model weights (_optional_, _default_ = 'weights/bengalimodal.h5')

`--image_size` - Size image at which the model was trained. (_optional_, _default_ = 64)

Predicted result will be saved in **result.csv**, in a format: _(image_name, head_root, head_vowel, head_consonant)_





### Training:
```
python  train.py 
```

#### Command line arguments:
`--n_epochs`  -  Number of epochs to train model. (_optional_, _default_ = 16)

`--n_parquets` - Number of parquets to train on. (_optional_, _default_ = 4)

`--batch_size` - Model batch size. (_optional_, _default_ = 256)

`--image_size` - Size image at which the model will be trained. (_optional_, _default_ = 64)

`--weights_path` - Path for saving model weights. (_optional_, _default_ = 'weights/bengalimodal.h5')


Script loads train data from: _train_data/competition_files_












