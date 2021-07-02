# Transfer Learning

- This section applies the transfer learning approach using the `Xception` architecture.

**Results**

- Accuracy on the test set = ~79.19%

*The [pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) library was used to load the Xception pre-trained model.*


---
**Logs**

- Output of the forward pass has the format shown in `output_model_raw.txt`.

- The saved model from the notebook in this folder can be found [here](https://drive.google.com/file/d/1x6SFqceZvOZrwHRPdJFVkOtjwS3ZM3LO/view?usp=sharing). It can be loaded (using eg: PyTorch) and can be used directly for inference.

- The initial analysis included training the pre-trained `ResNet50` model which yielded 73.91% accuracy on the test set. For storage purposes, the relevant notebook (`resnet50.ipynb`) has been included.
