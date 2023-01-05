import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # initial
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu0 = nn.ReLU()

        # block 1
        self.conv11 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu12 = nn.ReLU()
        self.shortcut12 = nn.Sequential()

        self.conv13 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu14 = nn.ReLU()
        self.shortcut14 = nn.Sequential()

        self.conv15 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(16, momentum=0.1)
        self.relu16 = nn.ReLU()
        self.shortcut16 = nn.Sequential()

        # block 2
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu22 = nn.ReLU()
        self.shortcut22 = LambdaLayer(lambda x:
                                      F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 32 // 4, 32 // 4), "constant", 0))

        self.conv23 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu23 = nn.ReLU()
        self.conv24 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu24 = nn.ReLU()
        self.shortcut24 = nn.Sequential()

        self.conv25 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu25 = nn.ReLU()
        self.conv26 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26 = nn.BatchNorm2d(32, momentum=0.1)
        self.relu26 = nn.ReLU()
        self.shortcut26 = nn.Sequential()

        # block 3
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu32 = nn.ReLU()
        self.shortcut32 = LambdaLayer(lambda x:
                                      F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 64 // 4, 64 // 4), "constant", 0))

        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu33 = nn.ReLU()
        self.conv34 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn34 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu34 = nn.ReLU()
        self.shortcut34 = nn.Sequential()

        self.conv35 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu35 = nn.ReLU()
        self.conv36 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn36 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu36 = nn.ReLU()
        self.shortcut36 = nn.Sequential()

        # final
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        # initial
        out0 = self.relu0(self.bn0(self.conv0(x)))

        # block 1
        out = self.relu11(self.bn11(self.conv11(out0)))
        out = self.bn12(self.conv12(out))
        out += self.shortcut12(out0)
        out12 = self.relu12(out)

        out = self.relu13(self.bn13(self.conv13(out12)))
        out = self.bn14(self.conv14(out))
        out += self.shortcut14(out12)
        out14 = self.relu14(out)

        out = self.relu15(self.bn15(self.conv15(out14)))
        out = self.bn16(self.conv16(out))
        out += self.shortcut16(out14)
        out16 = self.relu16(out)

        # block 2
        out = self.relu21(self.bn21(self.conv21(out16)))
        out = self.bn22(self.conv22(out))
        out += self.shortcut22(out16)
        out22 = self.relu22(out)

        out = self.relu23(self.bn23(self.conv23(out22)))
        out = self.bn24(self.conv24(out))
        out += self.shortcut24(out22)
        out24 = self.relu24(out)

        out = self.relu25(self.bn25(self.conv25(out24)))
        out = self.bn26(self.conv26(out))
        out += self.shortcut26(out24)
        out26 = self.relu26(out)

        # block 3
        out = self.relu31(self.bn31(self.conv31(out26)))
        out = self.bn32(self.conv32(out))
        out += self.shortcut32(out26)
        out32 = self.relu32(out)

        out = self.relu33(self.bn33(self.conv33(out32)))
        out = self.bn34(self.conv34(out))
        out += self.shortcut34(out32)
        out34 = self.relu34(out)

        out = self.relu35(self.bn35(self.conv35(out34)))
        out = self.bn36(self.conv36(out))
        out += self.shortcut36(out34)
        out36 = self.relu36(out)

        # final
        out = F.avg_pool2d(out36, out36.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out