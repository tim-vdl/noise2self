import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import skimage
import sklearn

from pathlib import Path
import matplotlib.pyplot as plt

from mask import Masker
from data_utils import NoisyDataset
from data_utils import train_test_split_dataset
from util import plot_tensors
from models.unet import Unet


def main():
	if torch.cuda.is_available:
		device = torch.device('cuda')
		print('Running on GPU')
	else:
		device = torch.device('cpu')
		print('Running on CPU')
	
	# Load dataset
	noisy_dataset = NoisyDataset(
		input_path = Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_2')
	)

	# Split in training and test dataset
	noisy_train_dataset, noisy_test_dataset = train_test_split_dataset(noisy_dataset, split=0.1)

	# Show image of the training set
	noisy = noisy_train_dataset[0]
	noisy= noisy.float()

	masker = Masker(width = 4, mode='interpolate')
	net_input, mask = masker.mask(noisy.unsqueeze(0), 0)
	plot_tensors(
		[mask, noisy, net_input[0], net_input[0] - noisy[0]],
		["Mask", "Noisy Image", "Neural Net Input", "Difference"]
		)
	plt.show()

	model = Unet().to(device)
	loss_function = MSELoss()
	optimizer = Adam(model.parameters(), lr=0.001)

	data_loader = DataLoader(noisy_train_dataset, batch_size=32, shuffle=True)

	epochs = 100
	for e in range(epochs):
		for i, batch in enumerate(data_loader):
			noisy_images = batch
			noisy_images = noisy_images.to(device)
			
			net_input, mask = masker.mask(noisy_images, i)
			net_output = model(net_input)
			
			loss = loss_function(net_output*mask, noisy_images*mask)
			
			optimizer.zero_grad()		
			loss.backward()
			optimizer.step()
			
			if i % 1 == 0:
				print(f"Epoch {e} | Loss  batch {i}: {round(loss.item(), 6)}")
	
	test_data_loader = DataLoader(
		noisy_test_dataset,
		batch_size=32,
		shuffle=False,
		num_workers=0
		)

	i, noisy_test_batch = next(enumerate(test_data_loader))
	output = model(noisy_test_batch.to(device))
	
	for idx in range(len(noisy_test_batch)):
		plot_tensors(
			[noisy_test_batch[idx], output[idx]],
			["Noisy Test Image", "Inference"]
			)
		plt.show()

if __name__ == '__main__':
	main()