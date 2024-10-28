import os, matplotlib.pyplot as plt
import torch 

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from SuperUtils import DataUtils, TrainUtils
from PIL import Image
from PitchLengthPredictionNet import Net


# sample = [ input, target ]
def segment_image(img_path, mode='RGB'):
	image = Image.open(img_path).convert(mode)
	img_width, img_height = image.size
	cell_width = img_width // 3
	cell_height = img_height // 3
	cropped_images = []
	for i in range(3):
		for j in range(3):
			left = j * cell_width
			upper = i * cell_height
			right = (j + 1) * cell_width
			lower = (i + 1) * cell_height
			cropped_image = image.crop((left, upper, right, lower))
			cropped_images.append(cropped_image)
	return cropped_images

def get_samples(path='data\\textures\\periodic_cholesteric', pitches=[15, 17, 19, 21, 23, 25, 27, 29]):
	samples = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if '.png' in file:
				filepath = f'{root}\\{file}'
				imgs = segment_image(filepath)
				lbl = float(root.split('=')[-1])
				lbl_tensor = torch.zeros(8)
				for idx, pitch in enumerate(pitches):
					if lbl == pitch:
						lbl_tensor[idx] = 1
						break
				for img in imgs:
					samples.append([img, lbl_tensor])
	return samples
				

config = {
	'loss_fn': 'MSELoss',
	'epochs': 10,
	'img_shape': (1, 200, 200),
	'device': 'cuda',
	'batch_size': 16,
	'lr': 0.001
		}

if config['img_shape'][0] == 3:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(config['img_shape'][1:]),
		])
else:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(config['img_shape'][1:]),
		transforms.Grayscale(config['img_shape'][0]),
		])	



samples = get_samples()
print(f"Len Samples: {len(samples)}")

train_samples, test_samples = samples[:int(0.75*len(samples))], samples[int(0.75*len(samples)):]
train_dataset, test_dataset = DataUtils.CustomDataset(train_samples, transforms=transform), DataUtils.CustomDataset(test_samples, transforms=transform)

train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle=True, drop_last=True)
net = Net(config=config, in_channels=1).to(config['device'])

trainer = TrainUtils.Trainer(config=config, net=net, trainloader=train_dataloader, testloader=test_dataloader)
trainer.train()