import torch
import torch.nn as nn
import torch.nn.functional as F
from option import args as argss


def sparseFunction(weight, threshold, gate=torch.sigmoid):
    res=None

    if argss.activation=='soft':
        res=torch.sign(weight) * torch.relu(torch.abs(weight) - gate(threshold))

    elif argss.activation == 'hard':
        res=weight*((torch.abs(weight)>gate(threshold)).float())

    elif argss.activation == 'dropout':
        mask=torch.rand(weight.shape).cuda()
        rate=argss.targetSKSSparsity/100
        mask[mask<rate]=0
        mask[mask>=rate]=1
        res=weight*mask

    return res


class IKSConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if argss.sparseMode == 'lw':
            self.threshold = nn.Parameter(argss.t0 * torch.ones(1, 1, 1, 1))
        elif argss.sparseMode == 'gw':
            self.threshold = nn.Parameter(argss.t0 * torch.ones(64, 1, 1, 1))
        elif argss.sparseMode == 'kw':
            self.threshold = nn.Parameter(argss.t0 * torch.ones(64, 64, 1, 1))
        elif argss.sparseMode == 'pw':
            if argss.model=='RCAN':
                self.threshold = nn.Parameter(argss.t0 * torch.ones(64, 64, 3, 3))
            elif argss.model=='EDSR':
                self.threshold = nn.Parameter(argss.t0 * torch.ones(256, 256, 3, 3))
            # non-learnable
            # self.sparseThreshold = argss.t0 * torch.ones(64, 64, 3, 3).cuda()

    def forward(self, x):
        sparsedWeight = sparseFunction(self.weight, self.threshold)
        x = F.conv2d(x, sparsedWeight,
                     self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def getSparsity(self):
        sparsedWeight = sparseFunction(self.weight, self.threshold)
        temp = sparsedWeight.detach().cpu()
        temp[temp != 0] = 1
        return (100 - temp.mean().item() * 100), temp.numel(), self.threshold.mean().item()

    def decayThreshold(self, changeValue):
        with torch.no_grad():
            self.threshold*=changeValue

    def setThreshold(self, changeValue):
        with torch.no_grad():
            self.threshold*=changeValue/self.threshold.item()

    def getSparsedWeight(self):
        sparsedWeight = sparseFunction(self.weight, self.threshold)
        return sparsedWeight

    def getThreshold(self):
        return self.threshold.item()