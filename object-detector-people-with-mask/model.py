import torch
import torch.nn as nn


class PeopleMaskModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super(PeopleMaskModel, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.adaptor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (input_size // 8) * (input_size // 8), 128),
            nn.ReLU(),
        )

        # Class (mask/no mask)
        self.classifier_head = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1),
        )

        # Bound box (x, y, w, h)
        self.regression_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)  # (batch_size, 64, input_size // 8, input_size // 8)
        x = self.adaptor(x)  # (batch_size, 64)
        return torch.cat((self.classifier_head(x), self.regression_head(x)), 1)
