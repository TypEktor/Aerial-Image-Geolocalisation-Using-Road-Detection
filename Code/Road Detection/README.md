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

![U-Net]([https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Object%20Detection/Images/YoloSystem.png](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Road%20Detection/Images/U-Net.png)?raw=true)

