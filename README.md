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

![](https://github.com/albellov/SR_kspace/blob/master/images/result.png?raw=true)
From top to bottom: *Bicubic, SRResNet, SRGAN, our*

From left to right: *X2 and X4 upsampling*


### Using

```
pip install -r requirements.txt
python setup.py install
```
