import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=10)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        #self.fc1 = nn.Linear(16280, 1000)
        self.fc1 = nn.Linear(11340, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=0)


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, dropout=0.15):
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_name, dropout)
        #fix 
        #self.fc1 = nn.Linear(38400, 512)
        if vgg_name == 'VGG_MINE':
            self.fc1 = nn.Linear(2048, 512)
        else:
            self.fc1 = nn.Linear(6144, 512)
        self.drop = nn.Dropout(p=0.55)
        self.fc2 = nn.Linear(512, num_classes)
    
    def _make_layers(self, vgg_name, drop):
        layers = []
        in_channels = 1
        cfg = {
            'VGG_MINE': [16, 'M', 'D', 32, 'M', 'D', 64, 'M', 'D', 128, 'M', 'D', 256 ],
            'VGG11': [64, 'D', 'M', 128, 'D', 'M', 256, 'D', 256, 'D', 'M', 512, 'D', 512, 'D', 'M', 512, 'D', 512, 'D', 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        for x in cfg[vgg_name]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(p=drop)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return F.log_softmax(out,dim=0)

