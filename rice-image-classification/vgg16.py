from torch import nn, flatten


def out_channel_size(in_channels, kernel_size, stride, padding):
    # Output shape of the convolutional layer equation
    return int(((in_channels - kernel_size + 2 * padding) / stride) + 1)


class VGG16(nn.Module):

    def __init__(self, num_channels, image_length, num_classes):
        """
        VGG16 model for image classification, based on Very Deep Convolutional
        Networks for Large-Scale Image Recognition paper.

        :param num_channels: 1 for gray-scale images, 3 for RGB images.
        :param image_length: length of one side of the square image in pixels.
        :param num_classes: the total number of classes in the dataset.
        """
        super(VGG16, self).__init__()

        # Layer 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # Layer 2
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # Layer 3
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # Layer 4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # Layer 5
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_length = out_channel_size(image_length, 2, 2, 0)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512 * image_length * image_length, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Softmax classifier
        self.softmax = nn.Sequential(
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # X input shape: (B, C, H, W) where B = batch size, C = num_channels, H = image_height, W = image_width
        # E.g.: (64, 3, 250, 250)

        # Layer 1
        x = self.conv1_1(x)  # (64, 64, 250, 250)
        x = self.conv1_2(x)  # (64, 64, 250, 250)
        x = self.pool1(x)    # (64, 64, 125, 125)

        # Layer 2
        x = self.conv2_1(x)  # (64, 128, 125, 125)
        x = self.conv2_2(x)  # (64, 128, 125, 125)
        x = self.pool2(x)    # (64, 128, 62, 62)

        # Layer 3
        x = self.conv3_1(x)  # (64, 256, 62, 62)
        x = self.conv3_2(x)  # (64, 256, 62, 62)
        x = self.conv3_3(x)  # (64, 256, 62, 62)
        x = self.pool3(x)    # (64, 256, 31, 31)

        # Layer 4
        x = self.conv4_1(x)  # (64, 512, 31, 31)
        x = self.conv4_2(x)  # (64, 512, 31, 31)
        x = self.conv4_3(x)  # (64, 512, 31, 31)
        x = self.pool4(x)    # (64, 512, 15, 15)

        # Layer 5
        x = self.conv5_1(x)  # (64, 512, 15, 15)
        x = self.conv5_2(x)  # (64, 512, 15, 15)
        x = self.conv5_3(x)  # (64, 512, 15, 15)
        x = self.pool5(x)    # (64, 512, 7, 7)

        # Fully connected layers
        x = flatten(x, 1)    # (64, 512 * 7 * 7) = (64, 25088)
        x = self.fc1(x)      # (64, 4096)
        x = self.fc2(x)      # (64, 4096)
        x = self.softmax(x)  # (64, 5)

        return x
