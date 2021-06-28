import os
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


DATASET_ROOT = "dataset_zurich/"
NEW_DATASET_ROOT = "data/"

def load_dataset():
	"""Arranges images classwise into separate directories."""

	if not os.path.isdir("data"):
		os.mkdir("data")
	if not os.path.isdir("data/noninteracting"):
		os.mkdir("data/noninteracting")
	if not os.path.isdir("data/merger"):
		os.mkdir("data/merger")

	src_train_merger = DATASET_ROOT + "merger/training"
	src_val_merger = DATASET_ROOT + "merger/validation"
	src_test_merger = DATASET_ROOT + "merger/test"
	src_train_noninteracting = DATASET_ROOT + "noninteracting/training"
	src_val_noninteracting = DATASET_ROOT + "noninteracting/validation"
	src_test_noninteracting = DATASET_ROOT + "noninteracting/test"

	dest_merger = NEW_DATASET_ROOT + "merger/"
	dest_noninteracting = NEW_DATASET_ROOT + "noninteracting/"

	for filename in os.listdir(src_train_merger):
	  shutil.move(os.path.join(src_train_merger, filename), dest_merger)
	for filename in os.listdir(src_val_merger):
	  shutil.move(os.path.join(src_val_merger, filename), dest_merger)
	for filename in os.listdir(src_test_merger):
	  shutil.move(os.path.join(src_test_merger, filename), dest_merger)

	for filename in os.listdir(src_train_noninteracting):
	  shutil.move(os.path.join(src_train_noninteracting, filename), dest_noninteracting)
	for filename in os.listdir(src_val_noninteracting):
	  shutil.move(os.path.join(src_val_noninteracting, filename), dest_noninteracting)
	for filename in os.listdir(src_test_noninteracting):
	  shutil.move(os.path.join(src_test_noninteracting, filename), dest_noninteracting)

	shutil.rmtree(DATASET_ROOT)


def num_examples():
	"""Classwise no. of files."""
	sizes = {}
	for class_ in os.listdir(NEW_DATASET_ROOT):
		class_path = os.path.join(NEW_DATASET_ROOT, class_)
		size = len(os.listdir(class_path))
		sizes[class_] = size
	return sizes


# TODO: Smart cropping of images (Ideas: Object detection -> Cut interesting sections)

def resize(img, size=100):
  """
  size: Final image size (Same for x and y).

  Notes
  -----
  100 is a random choice. Options could be to find the distribution of image sizes in the
  whole dataset and choose a value based on that.

  """
  resized_img = cv2.resize(img, (size, size))
  return resized_img


def create_data(img_size=100):
	"""Generate stacked flattened vectors where each vector is the 
	flattened representation of an image.
	
	Notes
	-----
	Takes ~30min to get data (shape = ((160000, 10000))), for example.
	"""
	# https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
	total_num_examples = sum(num_examples().values())
	data = np.empty((0, size*size))
	for class_ in os.listdir(NEW_DATASET_ROOT):
	  class_path = os.path.join(NEW_DATASET_ROOT, class_)
	  for image in os.listdir(class_path):
	    img = cv2.imread(os.path.join(class_path, image)).mean(axis=2)
	    img = resize(img)
	    img = img.flatten()
	    data = np.append(data, np.expand_dims(img, axis=0), axis=0)


	##### GENERATE LABELS FOR CLASSIFICATION #####
	sizes = num_examples()
	merger_labels = np.repeat(0, sizes["merger"])
	noninteracting_labels = np.repeat(1, sizes["noninteracting"])
	labels = np.hstack([merger_labels, noninteracting_labels]) # First write merger labels and then noninteracting because while creating "data", the merger directory gets traversed first, and then noninteracting.
	##### end #####

	return data, labels  # Return the flattened image vectors (data) and the corresponding labels (labels)


def dataset_to_numpy():
	"""Save numpy binary files of classwise images.

	Notes
	-----
	Files are ~2GB size!

	"""
	images = []
	for class_ in os.listdir(NEW_DATASET_ROOT):
		class_path = os.path.join(NEW_DATASET_ROOT, class_)
		for image in os.listdir(class_path):
			img = cv2.imread(os.path.join(class_path, image))
			images.append(np.array(img))
		images = np.array(images)
		np.save(f"{class_}.npy", images)


def split_data(img_pcs, labels, frac=0.75):
	"""Split data into training and testing sets.

	img_pcs: The principal components obtained after PCA.
		Corresponds to the first output of `apply_PCA` (i.e. image_pcs) function from data_transforms.py
	labels: Associated labels for the data.
		Corresponds to the second output of `create_data` (i.e. labels) function in this file.
	frac: Fraction of the dataset used. Must be the same value as frac argument of `normalize` function in data_transforms.py
	"""
	X = pd.DataFrame(img_pcs)
	y = pd.Series(labels)[:frac*len(X)]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size = 0.3, shuffle=True, random_state=1
	)  # shuffle=True is important since the data before splitting consists of all classes' examples together.

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":
	print(num_examples())
