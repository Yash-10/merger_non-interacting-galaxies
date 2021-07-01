# Transfer Learning

This section applies the transfer learning approach using the `Xception` architecture.

**Results**

- Accuracy on the test set = ~79.19%

*The [pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) library was used to load the Xception pre-trained model.*

The file `cpu_probabilities.txt` contains a list of exponentiated outputs of the model during **testing**. Note, they are not the direct outputs of the model. Refer to the notebook, if in doubt or for manipulation.

`output_model_raw.txt` is the actual output from model.

The saved model from the notebook in this folder can be found at: https://drive.google.com/file/d/1x6SFqceZvOZrwHRPdJFVkOtjwS3ZM3LO/view?usp=sharing. One can load it (using eg: PyTorch) and use it directly for inference.
