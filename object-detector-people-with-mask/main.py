import torch
from torch.utils.data import DataLoader

import dataset
import metrics
import model

# HYPERPARAMETERS
input_size = 244
learning_rate = 1e-4
num_epochs = 50
batch_size = 32
loss_class_weight = 0.4
channels = 3

# Load datasets
dataset_path = "datasets/obj/"
train_files, dev_files, val_files = dataset.list_files(dataset_path, shuffle=False, split_percentage=[80, 10])


def transformer(image, output):
    image = image / 255.0
    output = output.float()
    output[1:] = output[1:] / float(input_size)
    return image, output


train_dataset = dataset.PeopleMaskDataset(dataset_path, train_files, input_size=input_size,
                                          transform=transformer, channels=channels)
dev_dataset = dataset.PeopleMaskDataset(dataset_path, dev_files, input_size=input_size,
                                        transform=transformer, channels=channels)
val_dataset = dataset.PeopleMaskDataset(dataset_path, val_files, input_size=input_size,
                                        transform=transformer, channels=channels)

# Plot sample images
# image, output = train_dataset[0]
# dataset.plot_image(image, output[0], output[1:])
# image, output = dev_dataset[0]
# dataset.plot_image(image, output[0], output[1:])
# image, output = val_dataset[0]
# dataset.plot_image(image, output[0], output[1:])


# Model Training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on device: {device}")

model = model.PeopleMaskModel(input_size=input_size, num_classes=2, channels=channels)
model.to(device)
print("Number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_class = torch.nn.CrossEntropyLoss()
criterion_box = torch.nn.MSELoss()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

history = {
    "train_loss": [],
    "train_loss_class": [],
    "train_loss_box": [],
    "train_accuracy": [],
    "train_iou": [],
    "dev_loss": [],
    "dev_loss_class": [],
    "dev_loss_box": [],
    "dev_accuracy": [],
    "dev_iou": [],
}

for epoch in range(num_epochs):

    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = 0
    train_loss_class = 0
    train_loss_box = 0
    train_accuracy = 0
    train_iou = 0
    model.train()

    for batch in train_dataloader:
        if channels == 1:
            features = batch[0].to(device).unsqueeze(1)  # (batch_size, 1, input_size, input_size)
        else:
            features = batch[0].to(device).permute(0, 3, 1, 2)  # (batch_size, 3, input_size, input_size)
        output = batch[1].to(device)  # (batch_size, 5)

        # Forward pass
        prediction = model(features)

        # Calculate loss
        class_expected = output[:, 0].long()
        class_predicted = prediction[:, :2]
        loss_class = criterion_class(class_predicted, class_expected)

        box_expected = output[:, 1:]
        box_predicted = prediction[:, 2:]
        loss_box = criterion_box(box_predicted, box_expected)

        loss = loss_class_weight * loss_class + loss_box * (1 - loss_class_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_class += loss_class.item()
        train_loss_box += loss_box.item()
        train_accuracy += (class_predicted.argmax(1) == class_expected).float().sum().item()
        for i in range(len(output)):
            train_iou += metrics.intersection_over_union(box_predicted[i] * input_size, box_expected[i] * input_size)

    train_loss /= len(train_dataset)
    train_loss_class /= len(train_dataset)
    train_loss_box /= len(train_dataset)
    train_accuracy /= len(train_dataset)
    train_iou /= len(train_dataset)
    history["train_loss"].append(train_loss)
    history["train_loss_class"].append(train_loss_class)
    history["train_loss_box"].append(train_loss_box)
    history["train_accuracy"].append(train_accuracy)
    history["train_iou"].append(train_iou)
    print(f"Train Loss: {train_loss:.6f} | Class Loss: {train_loss_class:.6f} | Box Loss: {train_loss_box:.6f} | "
          f"Accuracy: {train_accuracy:.6f} | IoU: {train_iou:.6f}")

    # Validation
    model.eval()
    dev_loss = 0
    dev_loss_class = 0
    dev_loss_box = 0
    dev_accuracy = 0
    dev_iou = 0

    with torch.no_grad():
        for batch in DataLoader(dev_dataset, batch_size=batch_size):
            if channels == 1:
                features = batch[0].to(device).unsqueeze(1)  # (batch_size, 1, input_size, input_size)
            else:
                features = batch[0].to(device).permute(0, 3, 1, 2)  # (batch_size, 3, input_size, input_size)
            output = batch[1].to(device)  # (batch_size, 5)

            # Forward pass
            prediction = model(features)

            # Calculate loss
            class_expected = output[:, 0].long()
            class_predicted = prediction[:, :2]
            loss_class = criterion_class(class_predicted, class_expected)

            box_expected = output[:, 1:]
            box_predicted = prediction[:, 2:]
            loss_box = criterion_box(box_predicted, box_expected)

            loss = loss_class_weight * loss_class + loss_box * (1 - loss_class_weight)

            dev_loss += loss.item()
            dev_loss_class += loss_class.item()
            dev_loss_box += loss_box.item()
            dev_accuracy += (class_predicted.argmax(1) == class_expected).float().sum().item()
            for i in range(len(output)):
                dev_iou += metrics.intersection_over_union(box_predicted[i] * input_size, box_expected[i] * input_size)

    dev_loss /= len(dev_dataset)
    dev_loss_class /= len(dev_dataset)
    dev_loss_box /= len(dev_dataset)
    dev_accuracy /= len(dev_dataset)
    dev_iou /= len(dev_dataset)
    history["dev_loss"].append(dev_loss)
    history["dev_loss_class"].append(dev_loss_class)
    history["dev_loss_box"].append(dev_loss_box)
    history["dev_accuracy"].append(dev_accuracy)
    history["dev_iou"].append(dev_iou)
    print(f"Dev Loss: {dev_loss:.6f} | Class Loss: {dev_loss_class:.6f} | Box Loss: {dev_loss_box:.6f} | "
          f"Accuracy: {dev_accuracy:.6f} | IoU: {dev_iou:.6f}")

# Model Evaluation
with torch.no_grad():
    val_loss = 0
    val_loss_class = 0
    val_loss_box = 0
    val_accuracy = 0
    val_iou = 0

    for batch in DataLoader(val_dataset, batch_size=batch_size):
        if channels == 1:
            features = batch[0].to(device).unsqueeze(1)  # (batch_size, 1, input_size, input_size)
        else:
            features = batch[0].to(device).permute(0, 3, 1, 2)  # (batch_size, 3, input_size, input_size)
        output = batch[1].to(device)  # (batch_size, 5)

        # Forward pass
        prediction = model(features)

        # Calculate loss
        class_expected = output[:, 0].long()
        class_predicted = prediction[:, :2]
        loss_class = criterion_class(class_predicted, class_expected)

        box_expected = output[:, 1:]
        box_predicted = prediction[:, 2:]
        loss_box = criterion_box(box_predicted, box_expected)

        loss = loss_class + loss_box

        val_loss += loss.item()
        val_loss_class += loss_class.item()
        val_loss_box += loss_box.item()
        val_accuracy += (class_predicted.argmax(1) == class_expected).float().sum().item()
        for i in range(len(output)):
            val_iou += metrics.intersection_over_union(box_predicted[i] * input_size, box_expected[i] * input_size)

    val_loss /= len(val_dataset)
    val_loss_class /= len(val_dataset)
    val_loss_box /= len(val_dataset)
    val_accuracy /= len(val_dataset)
    val_iou /= len(val_dataset)
    print(f"Val Loss: {val_loss:.6f} | Class Loss: {val_loss_class:.6f} | Box Loss: {val_loss_box:.6f} | "
          f"Accuracy: {val_accuracy:.6f} | IoU: {val_iou:.6f}")


def inference(split, image, output):
    if channels == 1:
        image_fmt = image.unsqueeze(0).to(device).unsqueeze(1)  # add batch and channel dimensions
    else:
        image_fmt = image.to(device).unsqueeze(0).permute(0, 3, 1, 2)  # (batch_size, channels, input_size, input_size)
    prediction = model(image_fmt).cpu()

    class_predicted = prediction[:, :2].argmax(1).item()

    box_expected = output[1:] * input_size
    box_predicted = prediction[:, 2:].squeeze(0) * input_size

    image = image * 255.0
    image = image.type(torch.uint8)
    box_predicted = box_predicted.type(torch.int)
    box_expected = box_expected.type(torch.int)
    iou = metrics.intersection_over_union(box_predicted, box_expected).item()
    dataset.plot_image(image, class_predicted, box_predicted, box_expected, iou, split, channels)


with torch.no_grad():
    for i in torch.randint(0, len(train_dataset), (10,)):
        image, output = train_dataset[i]
        inference("train", image, output)

    for i in torch.randint(0, len(dev_dataset), (10,)):
        image, output = dev_dataset[i]
        inference("dev", image, output)

    for i in torch.randint(0, len(val_dataset), (10,)):
        image, output = val_dataset[i]
        inference("val", image, output)

"""
1-channel, AvgPool2d, loss_class_weight=0.4, lr=1e-3, epoch=50 (params 59,353,862)
Train Loss: 0.004576 | Class Loss: 0.011260 | Box Loss: 0.000121 | Accuracy: 0.957406 | IoU: 0.643346
Dev Loss: 0.007606 | Class Loss: 0.016572 | Box Loss: 0.001628 | Accuracy: 0.852713 | IoU: 0.447719
Val Loss: 0.018220 | Class Loss: 0.017749 | Box Loss: 0.000472 | Accuracy: 0.800000 | IoU: 0.440699

3-channel, AvgPool2d, loss_class_weight=0.4, lr=1e-3, epoch=50 (params 59,355,014)
Train Loss: 0.004147 | Class Loss: 0.010341 | Box Loss: 0.000019 | Accuracy: 0.989351 | IoU: 0.826650
Dev Loss: 0.005874 | Class Loss: 0.013981 | Box Loss: 0.000470 | Accuracy: 0.945736 | IoU: 0.630955
Val Loss: 0.014980 | Class Loss: 0.014775 | Box Loss: 0.000206 | Accuracy: 0.900000 | IoU: 0.629788

3-channel, AvgPool2d, loss_class_weight=0.2, lr=1e-3, epoch=50 (params 59,355,014)
Train Loss: 0.002131 | Class Loss: 0.010525 | Box Loss: 0.000033 | Accuracy: 0.983543 | IoU: 0.788433
Dev Loss: 0.003200 | Class Loss: 0.014500 | Box Loss: 0.000375 | Accuracy: 0.930233 | IoU: 0.597626
Val Loss: 0.013938 | Class Loss: 0.013719 | Box Loss: 0.000218 | Accuracy: 0.938462 | IoU: 0.591371

3-channel, AvgPool2d, loss_class_weight=0.4, lr=1e-4, epoch=50 (params 59,355,014)
Train Loss: 0.004156 | Class Loss: 0.010373 | Box Loss: 0.000011 | Accuracy: 0.988383 | IoU: 0.870508
Dev Loss: 0.005952 | Class Loss: 0.014183 | Box Loss: 0.000464 | Accuracy: 0.930233 | IoU: 0.635148
Val Loss: 0.014808 | Class Loss: 0.014585 | Box Loss: 0.000224 | Accuracy: 0.923077 | IoU: 0.621722
"""
