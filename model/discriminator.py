import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, stride=2, padding=1, kernel_size=4, scale_factor=32):
        super(Discriminator, self).__init__()
            # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second convolutional layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third convolutional layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth convolutional layer
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fifth convolutional layer (output layer)
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

        # Upsampling layer to resize the output to the input size
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.upsample(x)
        return x
    
# input_channels = 19
# model = Discriminator(in_channels=input_channels)
# summary(model, input_size=(19, 128, 64))