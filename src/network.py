import torchvision.models as models
import torch.nn as nn
import torch

device = torch.device('cuda:0')

class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg16_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16_bn(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg19(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Vgg19_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg19_bn(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
        self.net.features[0] = nn.Conv2d(3,64,3,padding=1)
    
    def forward(self, x):
        return self.net(x)

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18()
        self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=1)

    def forward(self,x):
        return self.net(x)

class Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet34()
        self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=512,out_features=1)

    def forward(self,x):
        return self.net(x)

class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet50()
        self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=2048,out_features=1)

    def forward(self,x):
        return self.net(x)

class Squeezenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.squeezenet1_0()
        #self.net.features[0] = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.classifier[1] = nn.Conv2d(512,1,1,stride=(1,1))

    def forward(self,x):
        return self.net(x)
    
class Densenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.densenet161()
        #self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.classifier = nn.Linear(in_features=2208,out_features=1)

    def forward(self,x):
        return self.net(x)

class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.inception_v3()
        #self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.fc = nn.Linear(in_features=2048,out_features=1)

    def forward(self,x):
        return self.net(x)

class Mobilenet_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.mobilenet_v3_large()
        #self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.classifier[3] = nn.Linear(in_features=1280,out_features=1)

    def forward(self,x):
        return self.net(x)

class Mobilenet_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.mobilenet_v3_small()
        self.net.conv1 = nn.Conv2d(3,64,7,stride=(2,2),padding=(3,3))
        self.net.classifier[3] = nn.Linear(in_features=1024,out_features=1)

    def forward(self,x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(2, 20)   
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))

def make_model(mn):
    if mn == 'Vgg16':
        net = Vgg16()
    elif mn == 'Vgg16-bn':
        net = Vgg16_bn()
    elif mn == 'Vgg19':
        net = Vgg19()
    elif mn == 'Vgg19-bn':
        net = Vgg19_bn()
    elif mn == 'Resnet18':
        net = Resnet18()
    elif mn == 'Resnet34':
        net = Resnet34()
    elif mn == 'Resnet50':
        net = Resnet50()
    elif mn == 'Squeezenet':
        net = Squeezenet()
    elif mn == 'Densenet':
        net = Densenet()
    elif mn == 'Inception':
        net = Inception()
    elif mn == 'Mobilenet-large':
        net = Mobilenet_large()
    elif mn == 'Mobilenet-small':
        net = Mobilenet_small()
            
    return net