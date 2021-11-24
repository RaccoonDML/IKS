import torch
import torch.nn as nn
import torch.nn.functional as F


def sparseFunction(weight, threshold, gate=torch.sigmoid):
    res=torch.sign(weight) * torch.relu(torch.abs(weight) - gate(threshold))
    return res


class IKSConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t0=-12
        #  PW as default
        self.threshold = nn.Parameter(t0 * torch.ones_like(self.weight))
        # non-learnable
        # self.sparseThreshold = argss.t0 * torch.ones(64, 64, 3, 3).cuda()

    def forward(self, x):
        sparsedWeight = sparseFunction(self.weight, self.threshold)
        x = F.conv2d(x, sparsedWeight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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

# reference: https://github.com/RAIVNLab/STR
