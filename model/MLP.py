import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        # 两层感知机
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.hidden1 = torch.nn.Linear(100,65)
        self.hidden2 = torch.nn.Linear(65,65)
        self.hidden3 = torch.nn.Linear(65,100)
        self.hidden4 = torch.nn.Linear(2*100,6)
    def forward(self,input):
        x = F.relu(self.hidden1(input))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        y = F.relu(self.hidden4(x))
        return y