import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import architecture
# Testing settings

parser = argparse.ArgumentParser(description='IMDN')
parser.add_argument("--dataset", type=str, default='Set5')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--save", type=str, default='dmltest')
parser.add_argument("--test_hr_folder", type=str, default='/home/mldai/workspace/IKS/SRDATA/dataset_fortest/HR',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='/home/mldai/workspace/IKS/SRDATA/dataset_fortest/LR/LRBI',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_x2.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

filepath = os.path.join(opt.test_hr_folder, opt.dataset, f'x{opt.upscale_factor}' )

filelist = os.listdir(filepath)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = architecture.IMDN(upscale=opt.upscale_factor)
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    # im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    # im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    # im_l = cv2.imread(os.path.join(opt.test_lr_folder,imname), cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB

    im_gt = cv2.imread(os.path.join(filepath, imname), cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
    im_l  = cv2.imread(os.path.join(opt.test_lr_folder,f'{opt.dataset}/x{opt.upscale_factor}', imname.replace('HR','LRBI')), cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]


    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)

    # cropped_sr_img = out_img
    # cropped_gt_img = im_gt

    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)


    # output_folder = os.path.join(opt.output_folder,
    #                              imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')
    output_folder = os.path.join(opt.output_folder,'{}/{}/x{}'.format(opt.save, opt.dataset, opt.upscale_factor))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    cv2.imwrite(os.path.join(output_folder,imname.replace('HR',opt.save)), out_img[:, :, [2, 1, 0]])
    print(i, imname)
    i += 1



print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
