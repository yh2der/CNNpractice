#!/usr/bin/env python
# coding: utf-8

# In[3]:


from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.block1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            ), 
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            ), 
            nn.Sequential( 
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(True)
            )
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), 
            nn.Flatten(), 
            nn.Linear(512, 10),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.block2(x) + x
        x = self.classifier(x)
        
        return x


# In[ ]:




