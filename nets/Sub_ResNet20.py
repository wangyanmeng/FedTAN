import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn.init as init

class BatchNorm2dInput(nn.Module):

    def __init__(self, num_features):
        super().__init__()

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)

        return input_.contiguous(), batchsize, channels, height, width


class BatchNorm2dMean1(nn.Module):

    def __init__(self, num_features):
        super().__init__()

    def forward(self, input_):
        numel = input_.size(1)
        mean1 = input_ / numel

        return mean1.contiguous()


class BatchNorm2dMean2(nn.Module):

    def __init__(self, num_features):
        super().__init__()

    def forward(self, mean1):
        mean = mean1.sum(1)

        return mean.contiguous()


class BatchNorm2dVar1(nn.Module):

    def __init__(self, num_features):
        super().__init__()

    def forward(self, input_, mean):
        bias_var1 = input_ - mean.unsqueeze(1)

        return bias_var1.contiguous()


class BatchNorm2dVar2(nn.Module):

    def __init__(self, num_features):
        super().__init__()

    def forward(self, bias_var1):
        numel = bias_var1.size(1)
        bias_var = (bias_var1.pow(2) / numel).sum(1)

        return bias_var.contiguous()


class BatchNorm2dModule11(nn.Module):

    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()

    def forward(self, input_, mean):
        batchsize, numel = input_.size()
        output1 = input_ - mean.unsqueeze(1)

        self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.detach()
        )

        return output1.contiguous()


class BatchNorm2dModule12(nn.Module):

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def forward(self, bias_var, batchsize):
        inv_std1 = bias_var + self.eps
        unbias_var = bias_var * batchsize / (batchsize - 1)
        self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * unbias_var.detach()
        )
        return inv_std1.contiguous()


class BatchNorm2dModule13(nn.Module):

    def __init__(self, num_features):
        super().__init__()

    def forward(self, output1, inv_std1):
        inv_std2 = 1 / inv_std1.pow(0.5)
        output2 = output1 * inv_std2.unsqueeze(1)

        return output2.contiguous()


class BatchNorm2dModule2(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, output2, batchsize, channels, height, width):
        output = output2 * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# basic blocks
# ---- mean in BN
class SubNet_Mean(nn.Module):
    def __init__(self, num_features):
        super(SubNet_Mean, self).__init__()
        self.bn_Mean1 = BatchNorm2dMean1(num_features)
        self.bn_Mean2 = BatchNorm2dMean2(num_features)

    def forward(self, input_):
        mean1 = self.bn_Mean1(input_)
        mean = self.bn_Mean2(mean1)
        return mean

# ---- variance in BN
class SubNet_Var(nn.Module):
    def __init__(self, num_features):
        super(SubNet_Var, self).__init__()
        self.bn_Var1 = BatchNorm2dVar1(num_features)
        self.bn_Var2 = BatchNorm2dVar2(num_features)

    def forward(self, input_, mean):
        bias_var1 = self.bn_Var1(input_, mean)
        bias_var = self.bn_Var2(bias_var1)
        return bias_var

# ---- (1) SubNet 1
class SubNet_1(nn.Module):
    def __init__(self, in_channels_num, out_channels_num, stride_num):
        super(SubNet_1, self).__init__()
        self.conv = nn.Conv2d(in_channels_num, out_channels_num, kernel_size=3, stride=stride_num, padding=1,
                              bias=False)
        self.bn_Input = BatchNorm2dInput(out_channels_num)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.conv(x)
        input_, batchsize, channels, height, width = self.bn_Input(out)
        return input_, batchsize, channels, height, width


# ---- (2) SubNet 2
class SubNet_2(nn.Module):
    def __init__(self, num_features, stride_num):
        super(SubNet_2, self).__init__()
        self.bn_11 = BatchNorm2dModule11(num_features)
        self.bn_12 = BatchNorm2dModule12(num_features)
        self.bn_13 = BatchNorm2dModule13(num_features)
        self.bn_2 = BatchNorm2dModule2(num_features)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(num_features, num_features, kernel_size=3, stride=stride_num, padding=1, bias=False)
        self.bn_Input = BatchNorm2dInput(num_features)
        self.apply(_weights_init)

    def forward(self, input_, mean, bias_var, batchsize, channels, height, width):
        output1 = self.bn_11(input_, mean)
        inv_std1 = self.bn_12(bias_var, batchsize)
        output2 = self.bn_13(output1, inv_std1)
        out = self.bn_2(output2, batchsize, channels, height, width)
        out = self.relu(out)
        out = self.conv(out)
        input_, batchsize, channels, height, width = self.bn_Input(out)
        return input_, batchsize, channels, height, width

# ---- (3) SubNet 3
# ---- method 1: without ResNet
class SubNet_3_woRes(nn.Module):
    def __init__(self, num_features):
        super(SubNet_3_woRes, self).__init__()
        self.bn_11 = BatchNorm2dModule11(num_features)
        self.bn_12 = BatchNorm2dModule12(num_features)
        self.bn_13 = BatchNorm2dModule13(num_features)
        self.bn_2 = BatchNorm2dModule2(num_features)
        self.relu = nn.ReLU()

    def forward(self, input_, mean, bias_var, batchsize, channels, height, width):
        output1 = self.bn_11(input_, mean)
        inv_std1 = self.bn_12(bias_var, batchsize)
        output2 = self.bn_13(output1, inv_std1)
        out = self.bn_2(output2, batchsize, channels, height, width)
        out = self.relu(out)
        return out


# ---- method 2: with ResNet, shortcut = Sequential()
class SubNet_3_ResSeq(nn.Module):
    def __init__(self, num_features):
        super(SubNet_3_ResSeq, self).__init__()
        self.bn_11 = BatchNorm2dModule11(num_features)
        self.bn_12 = BatchNorm2dModule12(num_features)
        self.bn_13 = BatchNorm2dModule13(num_features)
        self.bn_2 = BatchNorm2dModule2(num_features)
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, input_, mean, bias_var, out1, batchsize, channels, height, width):
        output1 = self.bn_11(input_, mean)
        inv_std1 = self.bn_12(bias_var, batchsize)
        output2 = self.bn_13(output1, inv_std1)
        out = self.bn_2(output2, batchsize, channels, height, width)
        out += self.shortcut(out1)
        out = self.relu(out)
        return out


# ---- method 3: with ResNet, shortcut = F.pad()
class SubNet_3_ResPad(nn.Module):
    def __init__(self, num_features):
        super(SubNet_3_ResPad, self).__init__()
        self.bn_11 = BatchNorm2dModule11(num_features)
        self.bn_12 = BatchNorm2dModule12(num_features)
        self.bn_13 = BatchNorm2dModule13(num_features)
        self.bn_2 = BatchNorm2dModule2(num_features)
        self.shortcut = LambdaLayer(lambda x:
                                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, num_features // 4, num_features // 4),
                                          "constant", 0))
        self.relu = nn.ReLU()

    def forward(self, input_, mean, bias_var, out2, batchsize, channels, height, width):
        output1 = self.bn_11(input_, mean)
        inv_std1 = self.bn_12(bias_var, batchsize)
        output2 = self.bn_13(output1, inv_std1)
        out = self.bn_2(output2, batchsize, channels, height, width)
        out += self.shortcut(out2)
        out = self.relu(out)
        return out


# ---- (4) fc
class SubNet_Fc(nn.Module):
    def __init__(self, num_features, num_classes):
        super(SubNet_Fc, self).__init__()
        self.origin = nn.Sequential()
        self.linear = nn.Linear(num_features, num_classes)
        self.apply(_weights_init)

    def forward(self, out):
        out = self.origin(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out