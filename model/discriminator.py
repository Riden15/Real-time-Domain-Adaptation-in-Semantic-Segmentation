import torch.nn as nn
from torchinfo import summary


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, stride=2, padding=1, kernel_size=4, scale_factor=32):
        super(Discriminator, self).__init__()
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=stride,
                      padding=padding),
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
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
        )

        # Upsampling layer to resize the output to the input size
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.upsample(x)
        return x


class DepthwiseDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels=1, stride=2, padding=1, kernel_size=4, scale_factor=32):
        super(DepthwiseDiscriminator, self).__init__()
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            # Depthwise convolutional
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Pointwise convolutional
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Second convolutional layer
            # Depthwise convolutional
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=64),
            nn.LeakyReLU(0.2, inplace=True),
            # Pointwise convolutional
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Third convolutional layer
            # Depthwise convolutional
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=128),
            nn.LeakyReLU(0.2, inplace=True),
            # Pointwise convolutional
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth convolutional layer
            # Depthwise convolutional
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=256),
            nn.LeakyReLU(0.2, inplace=True),
            # Pointwise convolutional
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Fifth convolutional layer (output layer)
            # Depthwise convolutional
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=512),
            # Pointwise convolutional
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1),
        )

        # Upsampling layer to resize the output to the input size
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.upsample(x)
        return x

# input_channels = 19
# model = Discriminator(in_channels=input_channels)
# model2 = DepthwiseDiscriminator(in_channels=input_channels)
# summary(model=model, 
#         input_size=(32, 19, 128, 64), # make sure this is "input_size", not "input_shape"
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )
# summary(model=model2, 
#         input_size=(32, 19, 128, 64), # make sure this is "input_size", not "input_shape"
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )
