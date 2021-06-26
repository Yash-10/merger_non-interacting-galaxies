import numpy as np
# from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def normalize(img_matrix, frac=0.75):
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
	image_pcs, explained_variances
		image_pcs will be used for classification as features.

	Notes
	-----
	n_components must be chosen such that significant amount of variance of data can be explained.
	For eg, n_components = 200 explains 81.98% of the variance. Common practice is to be able to explain ~95-99% of the variance. For this, increase `n_components`.

	"""
	explained_variances = {}
	num_components = [50, 100, 200, 300, 512]  # Values obtained after trial and error. Naturally, as `n_components` increase, the amount of variance explained increases.
	for num_component in num_components:
		pca = PCA(n_components=num_component)
		projected = pca.fit_transform(img_matrix)
		exp_var = np.sum(pca.explained_variance_)
		explained_variances[num_component] = exp_var

	image_pcs = pca.transform(img_matrix)  # Principal components obtained after PCA (To be used for classification)

	# print information about the PCA instance
	print(f"No. of samples: {pca.n_samples_}")
	print(f"No. of features: {pca.n_features_}")
	print(f"Max no. of components which can be used: {min(pca.n_samples_, pca.n_features_)}")

	np.save("images_pcs.npy", image_pcs)


	##### VISUALIZATION OF RECONSTRUCTED IMAGES AFTER PCA #####

	# Shape could be, for eg, (10000, 10000), where the first dimension is the number of examples
	# and the second one is the no. of elements in the flattened image data.
	# `imgs` are the reconstructed images after PCA analysis.
	imgs = pca.inverse_transform(image_pcs)  # Use the image PC's to construct images.

	# Show sample images
	to_show = 4
	index = np.random.choice(imgs.shape[0], to_show, replace=False)
	imgs_random = imgs[index]
	data_show = data[index]

	fig, ax = plt.subplots(nrows=to_show, ncols=2, figsize=(10, 20))
	for i in range(len(imgs_random)):
	  img = imgs_random[i].reshape(100, 100)  # Since all images are resized to 100 * 100. TODO: Make this robust.
	  data_show_img = data_show[i].reshape(100, 100)
	  ax[i, 0].imshow(img)
	  ax[i, 0].set_title("Image after PCA reduction")
	  ax[i, 1].imshow(data_show_img)
	  ax[i, 1].set_title("Image before PCA reduction")
	plt.show()

	##### end #####

	return image_pcs, explained_variances

	# pca = PCA(n_components=10)
	# transformed_data = pca.fit_transform(img_matrix)  # Shape could be, for eg, (10000, 10) if 10000 examples and 10 components.

	# explained_variance = np.sum(pca.explained_variance_)

	# return transformed_data, explained_variance