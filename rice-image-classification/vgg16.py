from torch import nn, flatten


def out_channel_size(in_channels, kernel_size, stride, padding):
    # Output shape of the convolutional layer equation
    return int(((in_channels - kernel_size + 2 * padding) / stride) + 1)


class VGG16(nn.Module):

    def __init__(self, num_channels, image_length, num_classes, dropout=0.5):
        """
        VGG16 model for image classification, based on Very Deep Convolutional
        Networks for Large-Scale Image Recognition paper.

        :param num_channels: 1 for gray-scale images, 3 for RGB images.
        :param image_length: length of one side of the square image in pixels.
        :param num_classes: the total number of classes in the dataset.
        """
        super(VGG16, self).__init__()

        # ConvLayer 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU())
        # ConvLayer 2
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # ConvLayer 3
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 4
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # ConvLayer 5
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 6
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 7
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # ConvLayer 8
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 9
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 10
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # ConvLayer 11
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 12
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        # ConvLayer 13
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * image_length * image_length, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Softmax classifier
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # X input shape: (B, C, H, W) where B = batch size, C = num_channels, H = image_height, W = image_width
        # E.g.: (64, 3, 250, 250)

        # Layer 1
        x = self.conv1_1(x)  # (64, 64, 250, 250)
        x = self.conv1_2(x)  # (64, 64, 250, 250)
        x = self.pool1(x)  # (64, 64, 125, 125)

        # Layer 2
        x = self.conv2_1(x)  # (64, 128, 125, 125)
        x = self.conv2_2(x)  # (64, 128, 125, 125)
        x = self.pool2(x)  # (64, 128, 62, 62)

        # Layer 3
        x = self.conv3_1(x)  # (64, 256, 62, 62)
        x = self.conv3_2(x)  # (64, 256, 62, 62)
        x = self.conv3_3(x)  # (64, 256, 62, 62)
        x = self.pool3(x)  # (64, 256, 31, 31)

        # Layer 4
        x = self.conv4_1(x)  # (64, 512, 31, 31)
        x = self.conv4_2(x)  # (64, 512, 31, 31)
        x = self.conv4_3(x)  # (64, 512, 31, 31)
        x = self.pool4(x)  # (64, 512, 15, 15)

        # Layer 5
        x = self.conv5_1(x)  # (64, 512, 15, 15)
        x = self.conv5_2(x)  # (64, 512, 15, 15)
        x = self.conv5_3(x)  # (64, 512, 15, 15)
        x = self.pool5(x)  # (64, 512, 7, 7)

        # Fully connected layers
        x = self.avgpool(x)  # (64, 512, 7, 7)
        x = flatten(x, 1)  # (64, 512 * 7 * 7) = (64, 25088)
        x = self.fc1(x)  # (64, 4096)
        x = self.fc2(x)  # (64, 4096)
        x = self.fc3(x)  # (64, 5)
        x = self.softmax(x)  # (64, 5)

        return x
