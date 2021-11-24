import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import architecture
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from model.IKSConv import IKSConv
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="/home/mldai/workspace/IKS/SRDATA/DIV2K",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

parser.add_argument("--save", type=str, default='dmltest')
parser.add_argument('--whichgpu', type=str, default='1')
parser.add_argument("--tar", type=float, default=50)

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.whichgpu

writter = SummaryWriter(log_dir='experiment/runs/' + args.save)

print("===> Loading datasets")

trainset = DIV2K.div2k(args)
# todo 3
testset = Set5_val.DatasetFromFolderVal("/home/mldai/workspace/IKS/SRDATA/benchmark/Urban100/HR",
                                       "/home/mldai/workspace/IKS/SRDATA/benchmark/Urban100/LR_bicubic/X{}/".format(args.scale),
                                       args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True

model = architecture.IMDN(upscale=args.scale)
l1_criterion = nn.L1Loss()

# todo:  calc params
TOT_param = sum(p.numel() for p in model.parameters())
IKS_thre_param = 0
for n, m in model.named_modules():
    if isinstance(m, IKSConv):
        IKS_thre_param += m.threshold.numel()
TOT_param -= IKS_thre_param
print(
    '\n*************\nOrigin_model parameters:\t{:.6f}M\nIKS_threshold parameters:\t{:.6f}M\nIKS_rate:\t\t\t{:.4f}\n*****************\n'.
    format(TOT_param / 1e6, IKS_thre_param / 1e6, IKS_thre_param / TOT_param))
################################################################################################

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=False)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)



def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_l1.item()))


def valid(epoch):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    avg_psnr/=len(testing_data_loader)
    avg_ssim/=len(testing_data_loader)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr, avg_ssim))
    writter.add_scalar("psnr", avg_psnr, epoch)
    psnr.append(avg_psnr)

    # todo Storing sparsity and threshold statistics for IKSConv models
    count = 0
    remain = 0.0
    for n, m in model.named_modules():
        if isinstance(m, IKSConv):
            sparsity_, total_params, thresh = m.getSparsity()
            remain += int(((100 - sparsity_) / 100) * total_params)
            count += total_params
    if count != 0:
        IKS_sparsity = (100 - (100 * remain / count))
        print("===> IKS_sparsity : {:.8f}".format(IKS_sparsity))
        writter.add_scalar("iks_sparsity", IKS_sparsity, epoch)
        sparsity.append(IKS_sparsity)
    # ***

    # todo save best model
    if count == 0:
        if max(psnr)==avg_psnr:
            save_checkpoint(epoch, 'model_best')
    else:
        global myBestPSE
        my_best = (sparsity[-1] > args.tar - 10 and psnr[-1] > myBestPSE[0])
        if my_best:
            myBestPSE = [psnr[-1], sparsity[-1], epoch]
            save_checkpoint(epoch, 'my_model_best')
        print(
            '(MyBest: {:.3f} and {:.2f}% @epoch {})'.format(
                myBestPSE[0],
                myBestPSE[1],
                myBestPSE[2]
            )
        )

    # # threshold decay ****************************************************
    D = 10
    K = 0.005
    tar = args.tar
    if tar != 0:
        global last_decay_epoch
        if epoch - last_decay_epoch > D and epoch < args.nEpochs - D:
            # Calculate S_cur: already in self.sparsity[-1]
            if abs(tar + 0.5 - IKS_sparsity) > 0.5:  # tar~tar+1
                last_decay_epoch = epoch
                decayRatio = 1 + K * (IKS_sparsity - tar)  # linear
                for name, param in model.named_modules():
                    if isinstance(param, IKSConv):
                        param.decayThreshold(decayRatio)
    # *******************************************************************************


def save_checkpoint(epoch, name):
    model_folder = "experiment/{}/".format(args.save)
    model_out_path = model_folder + "{}.pth".format(name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     # print(net)
#     print('Total number of parameters: %d' % num_params)


print("===> Training")
# print_network(model)
# todo global result
psnr=[]
sparsity=[]
last_decay_epoch=0
myBestPSE=[-1,-1,-1]

for epoch in range(args.start_epoch, args.nEpochs + 1):
    valid(epoch)
    train(epoch)
valid(args.nEpochs)