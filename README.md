# SR_kspace
k-space Deep Learning for Super-Resolution

### Authors
```
Aleksandr Belov, Mariia Kopylova, Olga Shtepa, Margarita Sharkova, Tamara Tsakhilova
```

### Abstract
One of the most demanded challenges is to reduce the effect of electromagnetic radiation during the MRI on humans and increase the speed of MRI. It could be achieved by decreasing the magnetic field, but would be received a low-resolution image. Then the main goal is formulated as finding the most optimal image recovery method solving the Super-Resolution problem. The performed methods in the study are CNN-based nets (SRGAN, SRResNet, our proposed net), and bicubic interpolation method.
The experiment were based [the Brain DICOM fastMRI dataset](https://fastmri.med.nyu.edu/).

The main difference our method from the others is training on *k*-space data. But *k*-space has a complex structure, and these are complex numbers in general. To work with the k-space, it was splitted into a real and imaginary part, fed to 2 channels, normalized by the values. 

Another feature is Generator loss is calculated in image-domain between SR and source HR image. That is back propagation passes through the FT and IFT layers.

To confirm the results of the study were compared MAE, PSNR, SSIM metrics of all of the approaches mentioned above. SSIM of the proposed method reaches the highest values of all of the others performed results. The model is also provides more realistic-looking images.

### Model

![](https://github.com/albellov/SR_kspace/blob/master/images/scheme_white.png?raw=true)


### Results

![](https://github.com/albellov/SR_kspace/blob/master/images/results.png?raw=true)

### Using

#### Install the environment
```
pip install -r requirements.txt
python setup.py install
```

#### Download and unpack the [data](https://yadi.sk/d/WjK5J8uL5R4YoQ) to `SR_kspace/data`
#### Train the model
```
python train.py [-h] [--channels CHANNELS] [--skip_connection SKIP_CONNECTION]
                [--bias BIAS] [--upscale_factor {2,4}] [--epochs EPOCHS]
                [--path_to_data PATH_TO_DATA] [--lr LR]
                [--batch_size BATCH_SIZE] [--random_state RANDOM_STATE]
                [--random_subset RANDOM_SUBSET] [--val_size VAL_SIZE]

Train Super Resolution Models

optional arguments:
  -h, --help            show this help message and exit
  --channels CHANNELS   Number of channels in Residual blocks
  --skip_connection SKIP_CONNECTION
                        Skip connection
  --bias BIAS           Bias in Conv layers for Generator
  --upscale_factor {2,4}
                        Super resolution upscale factor
  --epochs EPOCHS       Train epoch number
  --path_to_data PATH_TO_DATA
                        Path to data
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch size for train loader
  --random_state RANDOM_STATE
                        Random state
  --random_subset RANDOM_SUBSET
                        Size of subset for each epoch
  --val_size VAL_SIZE   Size of val set
  ```
