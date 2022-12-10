from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))

        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output

# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block = BasicBlock, num_blocks = [2,2,2,2], nClasses=6, in_dim = 128):
        super(ResNet, self).__init__()
        
        in_dim = in_dim
        n_layer = 1
        hidden_dim = 32
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=1)
        
        self.bi_gru1 = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional = True)
        self.bi_gru2 = nn.GRU(64, hidden_dim, n_layer, batch_first=True, bidirectional = True)
        
        # self.fc1 = TimeDistributed(nn.Linear(64, 32), True)
        # self.fc2 = TimeDistributed(nn.Linear(32, nClasses), True)
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, nClasses)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        (_, channel, seq_len, mel_bins) = x.shape
        x = x.view(-1, channel, seq_len, mel_bins)

        out = F.relu(self.bn1(self.conv1(x)))
        # out = out.to(torch.float32)
        out = self.dropout(out)

        out = self.layer1(out)
        out = F.max_pool2d(out, kernel_size=(1, 5), stride=(1, 5))
        out = self.dropout(out)
        
        out = self.layer2(out)
        out = F.max_pool2d(out, kernel_size=(1, 2), stride=(1, 2))
        out = self.dropout(out)
        
        out = self.layer3(out)
        out = F.max_pool2d(out, kernel_size=(1, 2), stride=(1, 2))
        out = self.dropout(out)
        
        out = self.layer4(out)
        out = F.max_pool2d(out, kernel_size=(1, 2), stride=(1, 2))
        out = self.dropout(out)
        
        out = out.permute(0, 2, 1, 3)
        out_need = out = out.reshape(out.shape[0], out.shape[1], -1)
        
        # out, _n = self.bi_gru1(out)
        # out = self.dropout(out)
        # out, _n = self.bi_gru2(out)
        # out_need = self.dropout(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out_need, out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2],in_dim=256)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3], in_dim=256)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3], in_dim = 1024)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
 
    
if __name__ == '__main__':
    model = ResNet50()
    summary(model, (2,256,80))
