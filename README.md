# Python command-line Bengali Classificator


## Getting Started

To download project:
```
git clone https://github.com/KopinHruk/cuddly-crankshaft.git 
```



## Usage without installation

### Prediction:
```
python predict.py  images_path  weights_path  image_size
```

#### Command line arguments:
`image_path`  -  Path to image or directory with images to predict on.

`weights_path` - Path for loading model weights (_optional_, _default_ = 'weights/bengalimodal.h5')

`image_size` - Size image at which the model was trained. (_optional_, _default_ = 64)






### Training:
```
python train.py  n_epochs  n_parquets  batch_size  image_size  weights_path
```

#### Command line arguments:
`n_epochs`  -  Number of epochs to train model. (_optional_, _default_ = 16)

`n_parquets` - Number of parquets to train on. (_optional_, _default_ = 4)

`batch_size` - Model batch size. (_optional_, _default_ = 256)

`image_size` - Size image at which the model will be trained. (_optional_, _default_ = 64)

`weights_path` - Path for saving model weights. (_optional_, _default_ = 'weights/bengalimodal.h5')











