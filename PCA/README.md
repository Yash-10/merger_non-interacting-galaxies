Using PCA to project images and then using it for dimensionality reduction.

**Sample Images before and after PCA (using 512 components)**
![PCA_before_after](images/pca_images.png)

`latest_latest_classification.ipynb` is done with 2048 PCA components on the whole dataset and takes around 45 min for fitting using `xgboost`. `latest_classification.ipynb` is done with 1024 components on the whole dataset with 1024 components and takes ~16-20 min for fitting.

**Observations**
- Increasing the components from 1024 to 2048 minutely decreases the accuracy (seen from the output in the respective notebooks) and using 2048 components even takes more than double the time taken by the 1024 approach. Hence, the best model is with using 1024 components, so far.
