import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

class STN3D(nn.Module):
    def __init__(self, input_channels=3):
        super(STN3D, self).__init__()
        self.input_channels = input_channels
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_channels * input_channels),
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = self.mlp1(x)
        x = F.max_pool1d(x, num_points).squeeze(2)
        x = self.mlp2(x)
        I = torch.eye(self.input_channels).view(-1).to(x.device)
        x = x + I
        x = x.view(-1, self.input_channels, self.input_channels)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(PointNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)
        return x


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(TargetEncoder, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc(x)
        return x

class Classification_Layer(nn.Module):
    def __init__(self, input_dim, num_class, use_bn=False):
        super(Classification_Layer, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, num_class)
        if (use_bn):
            self.bn1 = nn.BatchNorm1d(intermediate_layer)

    def forward(self, x, use_bn=False):
        if use_bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = self.fc1(x)

        return F.log_softmax(x, dim=1)

class ParamDecoder(nn.Module):
    def __init__(self, input_dim, intermediate_layer, embedding_size, use_bn=False):
        super(ParamDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, intermediate_layer)
        self.fc2 = nn.Linear(intermediate_layer, embedding_size)
        if (use_bn):
            self.bn1 = nn.BatchNorm1d(intermediate_layer)

    def forward(self, x, use_bn=False):
        if use_bn:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = self.fc1(x)
        x = self.fc2(x)

        return x

class ParamDecoder2(nn.Module):
    def __init__(self, input_dim, intermediate_layer, embedding_size, use_bn=False):
        super(ParamDecoder2, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, intermediate_layer)
        self.fc3 = nn.Linear(intermediate_layer, embedding_size)
        if (use_bn):
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(intermediate_layer)

    def forward(self, x, use_bn=False):
        if use_bn:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = self.fc1(x)
            x = self.fc2(x)

        x = self.fc3(x)

        return x

class LatentDecoder(nn.Module):
    def __init__(self, input_dim, embedding_size):
        super(LatentDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc_mu = nn.Linear(256, embedding_size)
        self.fc_sigma = nn.Linear(256, embedding_size)


    def forward(self, x, use_bn=False):
        x = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma

class TargetDecoder(nn.Module):
    def __init__(self, input_dim, num_points):
        super(TargetDecoder, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_points*3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = x.view(-1, self.num_points, 3)

        return x


class ECCV(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(ECCV, self).__init__()
        self.input_channels = input_channels
        self.stn1 = STN3D(input_channels)
        self.stn2 = STN3D(64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, embedding_size)
        self.fc_sigma = nn.Linear(256, embedding_size)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        x = x.transpose(2, 1)  # transpose to apply 1D convolution
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, num_points).squeeze(2)  # max pooling
        x = self.fc1(x)
        x = self.fc2(x)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma

### For Images ###

#
def set_parameter_requires_grad(model, is_fixed):
    if is_fixed:
        for param in model.parameters():
            param.requires_grad = False

    ## Set layer4 to trainable
    for name, param in model.layer4[0].named_parameters():
        param.requires_grad = True 
    for name, param in model.layer4[1].named_parameters():
        param.requires_grad = True 


class ImageEncoder(nn.Module):
    def __init__(self, embedding_size, is_fixed, use_pretrained=True):
        super(ImageEncoder, self).__init__()

        #ResNet18
        model_ft = models.resnet18(pretrained=use_pretrained)

        # Set trainable parameters
        set_parameter_requires_grad(model_ft, is_fixed)
    
        # Set output feature dim
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, embedding_size)
        
        input_size = 224    

        self.input_size = input_size
        self.model = model_ft

        # ###Debug
        # for name, param in model_ft.named_parameters():
        #     print(name)
        #     print(param.requires_grad)

        # exit()

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.model(x)

        return x






