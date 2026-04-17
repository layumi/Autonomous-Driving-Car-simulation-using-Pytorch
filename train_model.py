import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.0001)
        init.constant_(m.bias.data, 0.0)

class DriverNet(nn.Module):

    def __init__(self):
        super(DriverNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=100, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 3, 70, 320)
        output = self.conv_layers(x)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output



class ft_resnet18(nn.Module):
    def __init__(self):
        super(ft_resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential()
        linear_layers = nn.Sequential(nn.Linear(512, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1))
        linear_layers.apply(weights_init_kaiming)
        classifier = nn.Sequential(nn.Dropout(p=0.9), nn.Linear(128, 1))
        classifier.apply(weights_init_classifier)

        self.linear_layers = linear_layers 
        self.classifier = classifier

    def forward(self, x):
        x = x.view(x.size(0), 3, 70, 320)
        output = self.model(x)
        output = self.classifier(self.linear_layers(output))
        return output


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_resnet18()
    #net = DriverNet()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 70, 320))
    output = net(input)
    print('net output size:')
    print(output.shape)


