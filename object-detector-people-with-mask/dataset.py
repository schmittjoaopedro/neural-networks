import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


def list_files(dataset_dir, image_ext='.jpg', split_percentage=(70, 20), shuffle=False):
    files = []

    for r, d, f in os.walk(dataset_dir):
        for file in f:
            if file.endswith(".txt"):
                # first, let's check if there is only one object
                with open(dataset_dir + "/" + file, 'r') as fp:
                    lines = fp.readlines()
                    if len(lines) > 1:
                        continue

                strip = file[0:len(file) - len(".txt")]
                # secondly, check if the paired image actually exist
                image_path = dataset_dir + "/" + strip + image_ext
                if os.path.isfile(image_path):
                    files.append(strip)

    size = len(files)
    print(str(size) + " valid case(s)")

    if shuffle:
        random.shuffle(files)

    split_training = int(split_percentage[0] * size / 100)
    split_validation = split_training + int(split_percentage[1] * size / 100)

    return files[0:split_training], files[split_training:split_validation], files[split_validation:]


def plot_image(image, clazz, box, box_original=None, iou=None, split=None):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if isinstance(clazz, torch.Tensor):
        clazz = clazz.item()
    if isinstance(box, torch.Tensor):
        box = box.numpy()
    if box_original is not None and isinstance(box_original, torch.Tensor):
        box_original = box_original.numpy()
    img_title = "Mask" if clazz == 0 else "No mask"
    rgb_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    # Clazz = 0 -> with mask (red), Clazz = 1 -> without mask (green)
    cv.rectangle(rgb_image, box, (255 * (1 - clazz), 255 * clazz, 0), 2)
    if box_original is not None:
        cv.rectangle(rgb_image, box_original, (0, 0, 255), 2)
    plt.imshow(rgb_image)
    plt.axis("off")
    if split:
        img_title = f"{split} - {img_title}"
    if iou:
        img_title = f"{img_title} - IoU: {iou:.2f}"
    plt.title(img_title)
    plt.show()


class PeopleMaskImageLoader(Dataset):

    def __init__(self, dataset_dir, image_files, transform=None, input_size=244):
        self.dataset_dir = dataset_dir
        self.image_files = image_files
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = cv.imread(f"{self.dataset_dir}/{image_file}.jpg", cv.IMREAD_GRAYSCALE)

        with open(f"{self.dataset_dir}/{image_file}.txt", 'r') as annot:
            line = annot.readlines()[0]
            clazz = int(line[0])
            box = [float(x) for x in line[1:].split()]

        new_image, new_box = self._escale_image_and_box(image, box)
        output = torch.tensor([clazz] + new_box, dtype=torch.int)

        if self.transform:
            new_image, output = self.transform(new_image, output)

        return new_image, output

    def _escale_image_and_box(self, image, box):
        height, width = image.shape
        max_size = max(height, width)
        r = max_size / self.input_size
        new_width = int(width / r)
        new_height = int(height / r)
        new_size = (new_width, new_height)
        resized = cv.resize(image, new_size, interpolation=cv.INTER_LINEAR)
        new_image = torch.zeros((self.input_size, self.input_size), dtype=torch.uint8)
        new_image[0:new_height, 0:new_width] = torch.tensor(resized)

        x, y, w, h = box[0], box[1], box[2], box[3]
        new_box = [int((x - 0.5 * w) * width / r), int((y - 0.5 * h) * height / r), int(w * width / r),
                   int(h * height / r)]

        return new_image, new_box
