# Road Detection
Deep neural networks can be considered a reliable approach in order to deal with semantic
image segmentation problems. As already cited, the first task of this work is
to effectively extract roads from high-resolutional aerial images. This section focuses
on describing all the implemented deep neural networks used for achieving the
aforementioned task.
Specifically, three models have been used:
- U-Net
- Residual U-Net
- Multi-Residual U-Net

### U-Net
Fully convolutional neural network design makes them an efficient architecture for
pixel-wise semantic segmentation. These architectures have been crucial in a lot of
modern state of the art approaches. Fully convolutional neural networks have been further
improved to what is known as the U-Net (Ronneberger, Fischer, and Brox, 2015). Encoder-decoder architectures such as U-Net can achieve high accuracy using small
datasets. This happens because the fully-connected layer is replaced with a series of
up convolutions on the decoder side, which still has learnable parameters, but much
fewer than a fully-connected layer.

As illustrated in the below figure, the architecture of a U-Net resembles the capital letter
’U’. The architecture consists of three parts, the contraction, which is the left side of
the figure, the bottleneck, which is the bottom part, and the expansion section. By
expanding each of these parts and starting with the contraction section, someone could
see that it is made of several contraction blocks. The first block takes an image tile as
input and applies two 3*3 convolution layers and a 2*2 max pooling. Moving on
to the next blocks, the number of feature maps doubles as the architecture approach
the bottleneck. The reason for this is that by doing it, the architecture can extract
more advanced features and also reduce the size of feature maps. At the bottom of the
architectures lies the bottleneck, which borders on contraction and expansion section.
This part uses two 3*3 convolution layers followed by a 2*2 upsampling convolution
layer.

![U-Net](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Road%20Detection/Images/U-Net.png?raw=true)


### Residual U-Net

The second implementation for detecting roads from the given datasets is a semantic
segmentation neural network that combines the residual learning (He et al., 2015) with
the U-Net architecture. Zhang et al. in (Zhang, Liu, and Wang, 2018) proposed an
architecture which has been shown that it can outperform the initial U-Net by using
only 1/4 of its parameters. By combining the strengths of both U-Net and residual
neural networks, this architecture can make the training process easier, and it also
facilitates information propagation without degradation. In order to achieve these,
skip connections within a residual unit, and between low levels and high levels of the
network will be used, and at the same time, fewer parameters will be kept.

![RESU-Net](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Road%20Detection/Images/ResUNet.png?raw=true)

### Multi-Residual U-Net
The authors in (Ibtehaz and Rahman, 2020a) implement some improvement
over the initial U-Net architecture, which led to the MultiRes U-Net.

By taking inspiration from the Inception (Szegedy et al., 2014) family
networks, the authors proposed a new MultiRes block. This block has replaced the
convolutional layer pairs from the initial U-Net. Instead of
3 * 3, 5 * 5 and 7 * 7 convolutional layers the MultiRes block contains three 3 * 3.
This idea was borrowed by Szegedy et al. (Szegedy et al., 2014) and has been used
for decreasing the memory requirement. The output of the last two 3 * 3 convolutional
layers approximate the output of that of 5 * 5 and 7 * 7. Then the output of these
three layers is concatenated to extract the spatial features in different scales. Finally, a
residual path has also been added.

![MultiRESU-Net](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Road%20Detection/Images/MultiResUnet.png?raw=true)

## Implementation

First of all, sorry for the kinda messy code, this work has been implemented 2 years ago, and because of the weak documentation, it was hard for me today to remember all the details and rerun the whole process again. This implementation is not for new starters as the provided code requires good knowledge of google colab and also good knowledge of how to deal with images as data and how to set up and implement Deep Neural Networks.

As mentioned in previous sections, I won't be able to provide the dataset, used for this project, due to copyrights. However, if you have your own set of images, together with their masks you will be able to use the code with a few modifications. If you are here for the FCN implementations, you will be able to use them with no problem.

In case you use Google Colab don't forget to run if you have your data in your Google Drive:
```
from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive
```

Finally, the code is provided in .py format, instead of .ipynb. This is because the code is still in progress to be converted into an automated pipeline, instead of different .ipynb files. As mentioned earlier, with good knowledge of ML and Google Colab, someone will be able to load everything in Google Colab and run the code.

