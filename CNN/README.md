# CNN

This section applies a seven layer convolutional network with four convolutional layers and three affine layers at the end for classification.

- The advantage of CNNs is that no pre-processing is needed on the input images to the network.
- Classification accuracy on the test set: ~94.72%. This was achieved by:
   - Downsampling the images using bicubic interpolation (using `OpenCV`'s bicubic interpolation method), and
   - Using a Learning rate scheduler and EarlyStopping. Moreover, due to the early stopping approach, the number of epochs to train could be set to a higher value without any overfitting worries.

1. The file `CNN_es_lrs.ipynb` contains additional implementation of `EarlyStopping` and learning rate scheduler (`StepScheduler`) to decrease the learning rate as the training proceeds. This was done since as the training proceeds, the model converges towards optimal weights and shouldn't overshoot the local minimum.
2. The file `CNN_downsample.ipynb` applies downsampling of images before feeding them to the convolutional network. The downsampling processing was done as part of the model using `F.interpolate`. This was attempted in the hope to make the model robust to various background objects (for eg: stars) that are present in the dataset. This approach yielded slight improvements in terms of accuracy (~94.40%) on the test set. Using nearest-neighbor interpolation gives 94.02% classification accuracy, yielding slightly worser results than bicubic interpolation.
3. The file `CNN_downsample_2.ipynb` applies downsampling as a part of `transforms`, instead of applying it as a part of the model. This achieves 94.72% test accuracy. Although there is only a slight improvement compared to the results in `CNN_downsample.ipynb`, it implies ~50 images more are correctly classified.
4. The file `CNN_only_conv_relu.ipynb` uses the approach of applying `conv-relu-dropout` (in that order) till the image size becomes 1X1 at the end. However, the test set accuracy could only reach ~84.24%.

There is a hope to use smart interpolators to help CNNs ignore "background" objects and learn patterns in merger and non-interacting galaxies. However, over-preprocessing could confuse CNNs.
