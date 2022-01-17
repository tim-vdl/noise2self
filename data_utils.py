from pathlib import Path
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import json
from nd2reader import ND2Reader

def load_custom_config(path='custom_config.json'):
	with open(path) as f:
		custom_config = json.load(f)
	return custom_config

def get_files_in_folder(path, file_extension):
	if not isinstance(path, Path):
		path = Path(path)
	files = [f for f in path.iterdir() if f.is_file() and f.suffix == file_extension]
	return files


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20):
	"""Percentile-based image normalization."""
	mi = np.percentile(x,pmin,axis=axis,keepdims=True)
	ma = np.percentile(x,pmax,axis=axis,keepdims=True)
	return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20):
	x = (x - mi) / ( ma - mi + eps )
	if clip:
		x = torch.clip(x,0,1)
	return x


def normalize_minmse(x, target):
	"""Affine rescaling of x, such that the mean squared error to target is minimal."""
	cov = np.cov(x.flatten(),target.flatten())
	alpha = cov[0,1] / (cov[0,0]+1e-10)
	beta = target.mean() - alpha*x.mean()
	return alpha*x + beta

def normalize_mi_ma_images(images, channelwise=True):
	c, h, w = images.shape
	if channelwise:
		for i in range(c):
			images[i,:,:] = normalize(images[i,:,:])
	else:
		images = normalize(images)
	return images

def weighted_channel_merge(images):
	merged_image = np.sum(images, axis=0)
	merged_image /= images.shape[0]
	return merged_image


class NoisyDataset(Dataset):
	def __init__(self, input_path, files=None, file_extension='.png', transform=None, ):
		self.input_path = input_path
		if files:
			self.files = files
		else:
			self.files = [self.input_path.joinpath(f) for f in get_files_in_folder(self.input_path, file_extension=file_extension)]
		self.file_extension = file_extension,
		self.n_files = len(self.files)
		self.transform = transform
		
	def __len__(self):
		return len(self.files)
	
	def __getitem__(self, index):
		file_name = self.files[index]
		img = normalize(read_image(str(file_name)), clip=True).float()
		if self.transform:
			img = self.transform(img)
		return img


class Channel2ChannelDataset(Dataset):
	'''
	Channel2ChannelDataset

	Dataset that reads in multiple channels of an image and assigns one channel as the target
	and the other channels as input.
	'''
	def __init__(self, input_paths, target_path, input_files=None, target_files=None, file_extension='.png', transform=None, ):
		self.input_paths = [input_paths] if isinstance(input_paths, Path) else input_paths
		if input_files:
			self.input_files = input_files
		else:
			self.input_files = []
			for input_path in self.input_paths:
				self.input_files.append([input_path.joinpath(f) for f in get_files_in_folder(input_path, file_extension=file_extension)])
		self.target_path = target_path
		self.target_files = [self.target_path.joinpath(f) for f in get_files_in_folder(self.target_path, file_extension=file_extension)]
		self.file_extension = file_extension,
		self.n_files = len(self.input_files[0])
		self.transform = transform
		
	def __len__(self):
		return len(self.target_files)
	
	def __getitem__(self, index):
		imgs = ()
		for files in self.input_files:
			file_name = files[index]
			img = normalize(read_image(str(file_name)), clip=True).float()
			imgs = imgs + (img,)
		input_imgs = torch.vstack(imgs)

		target_file_name = self.target_files[index]
		target = normalize(read_image(str(target_file_name)), clip=True).float()
		if self.transform:
			input_imgs = self.transform(input_imgs)
			target = self.transform(target)
		return input_imgs, target


class RandomChannel2ChannelDataset(Dataset):
	'''
	RandomChannel2ChannelDataset

	Dataset that reads in multiple channels of an image and randomly assigns one channel as the target
	and the other channels as input.
	'''
	def __init__(self, input_paths, input_files=None, file_extension='.png', transform=None):
		self.input_paths = [input_paths] if isinstance(input_paths, Path) else input_paths
		if input_files:
			self.input_files = input_files
		else:
			self.input_files = []
			for input_path in self.input_paths:
				self.input_files.append([input_path.joinpath(f) for f in get_files_in_folder(input_path, file_extension=file_extension)])
		self.file_extension = file_extension,
		self.n_files = len(self.input_files[0])
		self.transform = transform
		
	def __len__(self):
		return self.n_files
	
	def __getitem__(self, index):
		imgs = ()
		for files in self.input_files:
			file_name = files[index]
			img = normalize(read_image(str(file_name)), clip=True).float()
			imgs = imgs + (img,)
		# input_imgs, target = train_test_split(imgs, test_size=1, shuffle=True)
		# target = target[0]
		input_imgs = imgs
		target = random.choice(imgs)
		
		input_imgs = torch.vstack(input_imgs)

		if self.transform:
			input_imgs = self.transform(input_imgs)
			target = self.transform(target)
		return input_imgs, target


def train_test_split_dataset(dataset, split=0.1, transform=None, random_state=None):
	'''Split dataset in training and testing sets
	
	Args:
		dataset (CustomDataset): Custom dataset that loads samples from a folder with images.
		split (float): Proportion of the dataset to include in the test set
		transform (None, tuple): Transforms to use when loading images from the splitted 
								 datasets (Default=None).
		random_state(int, RandomState instance, or None): Controls the shuffling applied \
						  to the data before applying the split. Pass an int for reproducible \
						  output across multiple function calls. Default = None \
						  (see sklearn.model_selection.train.train_test_split)
	'''
	assert isinstance(split, float) and split <= 1.
	assert transform is None or isinstance(transform, tuple)
	
	if transform is None:
		transform = (None, None)
	else:
		assert len(transform) == 2

	indices = np.arange(len(dataset))
	train_indices_files, test_indices_files = train_test_split(indices, test_size=split, shuffle=True, random_state=random_state)

	# Create the splitted datasets
	train_set = NoisyDataset(
		input_path = dataset.input_path,
		files = [dataset.files[i] for i in train_indices_files],
		file_extension = dataset.input_path,
		transform= dataset.transform
		)
	test_set = NoisyDataset(
		input_path = dataset.input_path,
		files = [dataset.files[i] for i in test_indices_files],
		file_extension = dataset.input_path,
		transform= dataset.transform
		)
	return train_set, test_set


def train_val_test_split_dataset(dataset, split, transform=None, random_state=None):
	'''Split dataset in training, validation and testing sets
	
	Uses train_test_split_dataset() function twice to split in 3 datasets.
	
	Args:
		dataset (CustomDataset): Custom dataset that loads samples from a folder with images.
		split (list, np.ndarray): Proportion of the dataset to include in the training \
			validation and test set
		transform (None, tuple): Transforms to use when loading images from the splitted 
								 datasets (Default=None).
		random_state(int, RandomState instance, or None): Controls the shuffling applied \
						  to the data before applying the split. Pass an int for reproducible \
						  output across multiple function calls \
						  (see sklearn.model_selection.train.train_test_split)
	'''
	assert isinstance(split, (list, np.ndarray)) and len(split) in (2, 3) \
		and all(num <= 1. for num in split)
	assert np.sum(split)==1
	assert transform is None or isinstance(transform, tuple)
	
	if transform is None:
		transform = (None, None, None)
	else:
		assert len(transform) in (2,3)
	val_size   = split[1]
	if len(split)==3:
		test_size = split[2]
		train_set, test_set = train_test_split_dataset(
			dataset, split=test_size,
			transform=(transform[0], transform[2])
			)
		train_set, val_set  = train_test_split_dataset(
			train_set,
			split=val_size/(1 - test_size),
			transform=(train_set.transform,transform[1]),
			random_state=random_state
			)
		return train_set, val_set, test_set
	else:
		train_set, test_set = train_test_split_dataset(
			dataset,
			test_size=val_size,
			transform=(transform[0], transform[1]),
			random_state=random_state
			)
		return train_set, test_set


def load_2D_nd2_img(file_path, bundle_axes='cyx'):
	with ND2Reader(str(file_path)) as images:
		if bundle_axes is not None:
			images.bundle_axes = 'cyx'
			images = images[0]
			return images
		return images