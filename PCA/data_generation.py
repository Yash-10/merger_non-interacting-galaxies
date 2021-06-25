import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt


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

	os.rmdir(DATASET_ROOT)  # TODO: Raises error if directory is not empty.


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
	Takes ~55s to get data (shape = ((160000, 10000))), for example.
	"""
	# https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
	total_num_examples = sum(num_examples().values())
	data = np.empty((total_num_examples, img_size*img_size))
	for class_ in os.listdir(NEW_DATASET_ROOT):
		class_path = os.path.join(NEW_DATASET_ROOT, class_)
		for i, image in enumerate(os.listdir(class_path)):
			img = cv2.imread(os.path.join(class_path, image)).mean(axis=2)
			img = resize(img)
			img = img.flatten()
			data[i] = img  # No need of any normalization here. Can use `normalize` function to do that.
	return data


def dataset_to_numpy():
	"""Save numpy binary files classwise.

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


if __name__ == "__main__":
	print(num_examples())