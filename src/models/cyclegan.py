import torch
import torch.nn as nn
import functools


def get_norm_layer(norm_layer):
	# Match official CycleGAN: InstanceNorm2d without running stats
	if norm_layer is nn.InstanceNorm2d:
		return lambda num_features: nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
	return norm_layer


class ResnetBlock(nn.Module):
	def __init__(self, dim: int, norm_layer: nn.Module, use_dropout: bool = False):
		super().__init__()
		conv_block = []
		p = 0
		conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0), get_norm_layer(norm_layer)(dim), nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]
		conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, padding=0), get_norm_layer(norm_layer)(dim)]
		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


class ResnetGenerator(nn.Module):
	def __init__(self, input_nc: int, output_nc: int, ngf: int = 64, n_blocks: int = 9, norm_layer: nn.Module = nn.InstanceNorm2d, use_dropout: bool = False):
		super().__init__()
		assert(n_blocks >= 0)
		model = []
		model += [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), get_norm_layer(norm_layer)(ngf), nn.ReLU(True)]
		# Downsampling
		n_downsampling = 2
		mult = 1
		for i in range(n_downsampling):
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), get_norm_layer(norm_layer)(ngf * mult * 2), nn.ReLU(True)]
			mult *= 2
		# ResNet blocks
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, get_norm_layer(norm_layer), use_dropout=use_dropout)]
		# Upsampling
		for i in range(n_downsampling):
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), get_norm_layer(norm_layer)(int(ngf * mult / 2)), nn.ReLU(True)]
			mult = int(mult / 2)
		model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input) 