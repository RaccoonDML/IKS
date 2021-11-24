# todo calc param and flops
import torch
import copy

from torchsummaryX import summary
import models.archs.PAN_arch as PAN_arch

import argparse
parser = argparse.ArgumentParser(description='EDSR and MDSR')
parser.add_argument('--scale', type=int, default=4)
args = parser.parse_args()

model = PAN_arch.PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=args.scale)

if args.scale==2:
	# input LR x2, HR size is 720p
	summary(model, torch.zeros((1, 3, 640, 360)))
elif args.scale==3:
	# input LR x3, HR size is 720p
	summary(model, torch.zeros((1, 3, 426, 240)))
elif args.scale==4:
	# input LR x4, HR size is 720p
	summary(model, torch.zeros((1, 3, 320, 180)))

