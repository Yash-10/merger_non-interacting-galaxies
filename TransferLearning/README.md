# Transfer Learning

- This section applies the transfer learning approach using the pre-trained `ResNet` and `Xception` architectures.

**Results**

- Accuracy on the test set = ~94.75%

---
**ResNet**
- `ResNet34` model gives 94.747% accuracy on the test set, slightly higher than `ResNet18` (92.89%). Both the models seemed to overfit quickly after the 15 epoch mark. Hence, the accuracy of these models on the test set might increase by a more rigourous early stopping approach.
- Also, using the ResNet models showed sudden decrease in validation accuracy in one or two intermediate epochs.

**Xception**
- The saved model from the `TransferLearning.ipynb` notebook can be found [here](https://drive.google.com/file/d/1x6SFqceZvOZrwHRPdJFVkOtjwS3ZM3LO/view?usp=sharing). It can be loaded (using eg: PyTorch) and can be used directly for inference.

- The initial analysis included training the pre-trained `ResNet50` model which yielded 73.91% accuracy on the test set. For storage purposes, the relevant notebook (`resnet50.ipynb`) has been included.

- `SGD + momentum` optimizer (`momentum = 0.9`) gave slightly sub-optimal classification accuracy (74.13%) compared to when `Adam` optimizer was used (79.19%) keeping the other network parameters and other hyperparameters unchanged.
- Output of the forward pass has the format shown in `output_model_raw.txt` (for Xception).

---
The training and testing sets from the original dataset were combined and then splitted into new train and val folders using stratified sampling.

### References
*The [pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) library was used to load the Xception pre-trained model in `PyTorch`.*

---
- Adding center-cropping data augmentation and using early stopping + learning rate scheduler plays a major role in the performance of pre-trained models.
