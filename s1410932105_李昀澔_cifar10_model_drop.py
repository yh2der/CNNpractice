from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.dropout3 = nn.Dropout2d(0.2)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(256 * 4 * 4, 1024)
        self.dropout4 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)  
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)  
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x) 
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout4(x) 
        x = F.relu(self.linear1(x))
        x = self.dropout4(x) 
        x = F.log_softmax(self.linear2(x), dim=1)
        return x
