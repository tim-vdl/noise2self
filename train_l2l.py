import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import skimage
import sklearn

from pathlib import Path
import matplotlib.pyplot as plt

from mask import Masker
from data_utils import RandomChannel2ChannelDataset
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
	channel2channel_dataset = RandomChannel2ChannelDataset(
		input_paths = [
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_2'),
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_3'),
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_4'),
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_5'),
		]
	)

	# Show image of the training set
	inputs, target = channel2channel_dataset[0]

	plot_tensors(
		[inputs[:3,:,:], target],
		["Input", "Targets"]
		)
	plt.show()

	model = Unet(
		n_channel_in=len(channel2channel_dataset.input_paths),
		n_channel_out=1,
		residual=True
		).to(device)
	loss_function = L1Loss() # MSELoss()
	optimizer = Adam(model.parameters(), lr=0.0002)

	data_loader = DataLoader(channel2channel_dataset, batch_size=32, shuffle=True)

	epochs = 100
	writer = SummaryWriter()
	for e in range(epochs):
		running_loss = 0
		for b, batch in enumerate(data_loader):
			inputs, target = batch
			inputs, target = inputs.to(device), target.to(device)
			
			net_output = model(inputs)
			
			loss = loss_function(net_output, target)
			running_loss += loss.item()
			
			optimizer.zero_grad()		
			loss.backward()
			optimizer.step()
			
			if b % 1 == 0:
				print(f"Epoch {e+1} | Loss  batch {b+1}: {round(loss.item(), 6)}")
		average_loss_epoch = running_loss/(b +1)
		writer.add_scalar("Loss/train", average_loss_epoch, e)
		print(f"Epoch {e+1} | Loss: {round(average_loss_epoch, 6)}")
	writer.flush()

	channel2channel_test_dataset = RandomChannel2ChannelDataset(
		input_paths = [
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_2'),
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_3'),
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_4'),
			Path(r'D:\Data_TimVDL\Images\NikonHTM_1321N1_vs_SHSY5Y\tif_channels\channel_5')
		]
	)

	test_data_loader = DataLoader(
		channel2channel_test_dataset,
		batch_size=32,
		shuffle=False,
		num_workers=0
		)

	_, (test_input, test_target) = next(enumerate(test_data_loader))
	output = model(test_input.to(device))
	
	for idx in range(len(test_input)):
		input = test_input[idx]
		input_average = input.mean(0)
		input_sted = input.std(0)
		plot_tensors(
			[input[:3,:,:], output[idx], input_average, input_sted],
			["Input", "Inference", "Input average", "Input STED"]
			)
		plt.show()

if __name__ == '__main__':
	main()