# CNN

This section applies a seven layer convolutional network with four convolutional layers and three affine layers at the end for classification.

- The advantage of CNNs is that no pre-processing is needed on the input images to the network.
- Classification accuracy on the test set: ~93.99%

The file `CNN_es_lrs.ipynb` contains additional implementation of `EarlyStopping` and learning rate scheduler (`StepScheduler`) to decrease the learning rate as the training proceeds. This was done since as the training proceeds, the model converges towards optimal weights and shouldn't overshoot the local minimum.
