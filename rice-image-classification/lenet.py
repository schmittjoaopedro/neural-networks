from torch import nn, flatten


def out_channel_size(in_channels, kernel_size, stride, padding):
    # Output shape of the convolutional layer equation
    return int(((in_channels - kernel_size + 2 * padding) / stride) + 1)


class LeNet(nn.Module):

    def __init__(self, num_channels, image_width, image_height, num_classes):
        """
        LeNet model for image classification. Architecture proposed by Yann LeCun in 1998.

        :param num_channels: 1 for gray-scale images, 3 for RGB images.
        :param image_width: image width in pixels.
        :param image_height: image height in pixels.
        :param num_classes: the total number of classes in the dataset.
        """
        super(LeNet, self).__init__()

        # Initialize the first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        image_width = out_channel_size(image_width, 5, 1, 0)
        image_height = out_channel_size(image_height, 5, 1, 0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_width = out_channel_size(image_width, 2, 2, 0)
        image_height = out_channel_size(image_height, 2, 2, 0)

        # Initialize the second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
                               kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        image_width = out_channel_size(image_width, 5, 1, 0)
        image_height = out_channel_size(image_height, 5, 1, 0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        image_width = out_channel_size(image_width, 2, 2, 0)
        image_height = out_channel_size(image_height, 2, 2, 0)

        # Initialize only fully connected layer
        self.fc1 = nn.Linear(in_features=50 * image_width * image_height, out_features=500)
        self.relu3 = nn.ReLU()

        # Softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # X input shape: (B, C, H, W) where B = batch size, C = num_channels, H = image_height, W = image_width
        # E.g.: (2, 3, 250, 250)

        # First Layer
        x = self.conv1(x)  # (2, 20, 246, 246)
        x = self.relu1(x)  # (2, 20, 246, 246)
        x = self.maxpool1(x)  # (2, 20, 123, 123)

        # Second Layer
        x = self.conv2(x)  # (2, 50, 119, 119)
        x = self.relu2(x)  # (2, 50, 119, 119)
        x = self.maxpool2(x)  # (2, 50, 59, 59)

        # Third Layer
        x = flatten(x, 1)  # (2, 50 * 59 * 59) = (2, 174050)
        x = self.fc1(x)  # (2, 500)
        x = self.relu3(x)  # (2, 500)

        # Fourth Layer
        x = self.fc2(x)  # (2, 5)
        x = self.softmax(x)  # (2, 5)

        return x
