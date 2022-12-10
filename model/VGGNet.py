import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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

class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio = 4):
        
        super(VggishConvBlock, self).__init__()
        
        inter_channels = out_channels // reduction_ratio
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=inter_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=inter_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=True)
                              
        self.bn1 = nn.BatchNorm2d(inter_channels)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        

        
    # def init_weights(self):
    #
    #     init_layer(self.conv1)
    #     init_layer(self.conv2)
    #     init_bn(self.bn1)
    #     init_bn(self.bn2)
        
    def forward(self, input):
        
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        
        return x
    
    
class Vggish(nn.Module):
    def __init__(self, nClasses = 6):
        
        super(Vggish, self).__init__()
        
        in_dim = 1024
        n_layer = 1
        hidden_dim = 32
        
        self.conv_block1 = VggishConvBlock(in_channels=2, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock(in_channels=256, out_channels=512)
        
        # self.bi_gru1 = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional = True)
        # self.bi_gru2 = nn.GRU(64, hidden_dim, n_layer, batch_first=True, bidirectional = True)
        #
        # self.fc1 = TimeDistributed(nn.Linear(64, 32), True)
        # self.fc2 = TimeDistributed(nn.Linear(32, nClasses), True)
        #
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, nClasses)
        self.dropout = nn.Dropout(0.5)


        

    def forward(self, input, return_bottleneck=False):
        # (_, seq_len, mel_bins) = input.shape

        # x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''
        (_, channel, seq_len, mel_bins) = input.shape
        x = input.view(-1, channel, seq_len, mel_bins)

        x = self.conv_block1(x)
        x = F.max_pool2d(x, kernel_size=(1, 5), stride=(1, 5))
        x = self.dropout(x)
#         x = self.att1(x)
        x = self.conv_block2(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout(x)
#         x = self.att2(x)
        x = self.conv_block3(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout(x)
#         x = self.att3(x)
        x = self.conv_block4(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        x = self.dropout(x)
        ## print(x.shape)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = self.att4(x)
        ## print(x.shape)

        out = self.fc1(x)
        out_need = self.dropout(out)
        out = self.fc2(out_need)
        out = torch.sigmoid(out)
        return out_need, out
    
if __name__ == '__main__':
    model = Vggish()
    summary(model, (2,256, 80))
