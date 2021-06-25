import numpy as np
# from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def normalize(img_matrix, frac=0.6):
	"""Standardize data i.e. zero mean and unit variance for PCA.

	Parameters
	----------
	img_matrix: np.array
		NumPy array of flattened image vectors

	"""
	# Below commented approach blowed up RAM usage.
	# scaler = preprocessing.StandardScaler()
	# scaler.fit(img_matrix)
	# scaled_img_matrix = scaler.transform(img_matrix)
	# imgs_flattened_normalized = scaled_img_matrix.flatten()

	# return imgs_flattened_normalized

	img_matrix = img_matrix[:frac*len(img_matrix)]  # Select a subset to prevent RAM blowup
	img_matrix = (img_matrix - img_matrix.mean(axis=0)) / img_matrix.std()

	variance = np.sum(np.std(img_matrix, axis = 0) ** 2)  # Variance of data just before applying PCA

	return img_matrix, variance


def apply_PCA(img_matrix, n_components=10):
	"""Apply PCA on data.

	img_matrix: np.array
		The standardized data (without labels). The data is standardized using the `normalize` function.
		`img_matrix` here corresponds to `image_matrix` from `normalize`.

	Return
	------
	transformed_data, explained_variance
		transformed_data will be used for classification as features.

	Notes
	-----
	n_components must be chosen such that significant amount of variance of data can be explained.
	For eg, n_components = 200 explains 83.21% of the variance. Common practice is to be able to explain ~95-99% of the variance. For this, increase `n_components`.

	"""
	pca = PCA(n_components=10)
	transformed_data = pca.fit_transform(img_matrix)  # Shape could be, for eg, (10000, 10) if 10000 examples and 10 components.

	explained_variance = np.sum(pca.explained_variance_)

	return transformed_data, explained_variance


def visualize_PCA_data(transformed_data):
	# Shape could be, for eg, (10000, 10000), where the first dimension is the number of examples
	# and the second one is the no. of elements in the flattened image data.
	# `imgs` are the reconstructed images after PCA analysis.
	imgs = pca.inverse_transform(transformed_data)

	# Show sample images
	to_show = 4
	index = np.random.choice(imgs.shape[0], to_show, replace=False)
	imgs_random = imgs[index]

	for img in imgs_random:
		img = img.reshape(100, 100)  # Since all images are resized to 100 * 100. TODO: Make this robust.
		plt.imshow(img)
		plt.show()
