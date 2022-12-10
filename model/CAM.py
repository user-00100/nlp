import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from cbam import CBAM

class attCNN(nn.Module):
    # 
    def __init__(self, nClasses = 6):
        super(attCNN, self).__init__()
        
        
        in_dim = 256
        n_layer = 2
        hidden_dim = 32
        
        
        # Input(2, 256, 40)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.bam1 = CBAM(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.bam2 = CBAM(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.activation3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.bam3 = CBAM(128)
        """self.bi_gru1 = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional = True)
        self.bi_gru2 = nn.GRU(64, hidden_dim, n_layer, batch_first=True, bidirectional = True)"""
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, nClasses)

    def forward(self, x, return_bottleneck=False):
        
        (_, channel, seq_len, mel_bins) = x.shape
        x = x.view(-1, channel, seq_len, mel_bins)
        # print(x.shape)
        out = self.dropout1(F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=(1, 5), stride=(1, 5)))
        out = self.bam1(out)
        out = self.dropout2(F.max_pool2d(F.relu(self.bn2(self.conv2(out))), kernel_size=(1, 2), stride=(1, 2)))
        out = self.bam2(out)
        out = self.dropout3(F.max_pool2d(F.relu(self.bn3(self.conv3(out))), kernel_size=(1, 2), stride=(1, 2)))
        out = self.bam3(out)
        
        # print(out.shape)
        # out = out.permute(0, 2, 1, 3)
        # out = out.reshape(out.shape[0], out.shape[1], -1)
        # print(out.shape)
        """out = self.dropout1(out)
        out, _n = self.bi_gru1(out)
        #print(out.shape)
        out = self.dropout1(out)
        out, _n = self.bi_gru2(out)"""
        # print(out.shape)
        # print(_n.shape)
        # print(out.shape)


        (out, _) = torch.max(self.bn3(out), dim=-1)
        out = out.permute(0, 2, 1)
        need_data = out
        out = self.fc1(out)
        out = self.dropout1(out)

        
        out = self.fc2(out)
        # print(out.shape)
        out = torch.sigmoid(out)
        
        return need_data, out
    
    
if __name__ == '__main__':
    model = attCNN()
    # summary(model, (2, 256, 40))
    summary(model, (2, 256, 80))
