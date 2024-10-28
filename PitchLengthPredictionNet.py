'''
Based on
https://doi.org/10.1038/s41598-020-63662-9
'''
import torch, torch.nn as nn


class Block(nn.Module):
	def __init__(self, in_channels, out_channels, conv_kernel, pool_kernel):
		super().__init__()
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel)
		self.pool = nn.MaxPool2d(pool_kernel)
	def forward(self, x):
		x = self.relu(self.pool(self.conv(x)))
		return x


class Net(nn.Module):
	def __init__(self, config, in_channels=3, block_channel_sequence=(5, 5, 5, 5), conv_kernel=2, pool_kernel=2):
		super().__init__()
		self.relu = nn.ReLU()
		c, h, w = config['img_shape']
		self.out_channels = block_channel_sequence[-1]
		self.h, self.w = h, w
		nb = len(block_channel_sequence)
		self.nb = nb
		self.pool_kernel = pool_kernel 

		for n in range(nb):
			if n+1 == 1:
				setattr(self, f"block{n+1}", Block(in_channels, block_channel_sequence[n], conv_kernel, pool_kernel))
			else:
				setattr(self, f"block{n+1}", Block(block_channel_sequence[n-1], block_channel_sequence[n], conv_kernel, pool_kernel))

		self.final_h, self.final_w = self.check_shape(torch.rand(1, c, h, w))
		self.fc1 = nn.Linear(self.out_channels * self.final_h * self.final_w, 16)
		self.fc_out = nn.Linear(16, 8)

	@torch.inference_mode()
	def check_shape(self, x):
		for n in range(self.nb):
			x = getattr(self, f"block{n+1}")(x)
		return x.shape[2:]

	def forward(self, x):
		for n in range(self.nb):
			x = getattr(self, f"block{n+1}")(x)
			
		x = x.view(-1, self.out_channels * self.final_h * self.final_w)
		x = self.relu(self.fc1(x))
		return self.fc_out(x)



		
