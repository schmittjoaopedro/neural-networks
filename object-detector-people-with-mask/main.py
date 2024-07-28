import torch
from torch.utils.data import DataLoader

import dataset
import model
import metrics

# HYPERPARAMETERS
input_size = 244
learning_rate = 1e-3
num_epochs = 50
batch_size = 32

# Load datasets
dataset_path = "datasets/obj/"
train_files, dev_files, val_files = dataset.list_files(dataset_path, shuffle=False)


def transformer(image, output):
    image = image / 255.0
    output = output.float()
    output[1:] = output[1:] / float(input_size)
    return image, output


train_dataset = dataset.PeopleMaskImageLoader(dataset_path, train_files, input_size=input_size, transform=transformer)
dev_dataset = dataset.PeopleMaskImageLoader(dataset_path, dev_files, input_size=input_size, transform=transformer)
val_dataset = dataset.PeopleMaskImageLoader(dataset_path, val_files, input_size=input_size, transform=transformer)

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

model = model.PeopleMaskModel(input_size=input_size, num_classes=2)
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
        features = batch[0].to(device).unsqueeze(1)  # (batch_size, 1, input_size, input_size)
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

        loss = loss_box + loss_class

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
    print(f"Loss: {train_loss:.6f} | Class Loss: {train_loss_class:.6f} | Box Loss: {train_loss_box:.6f} | "
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
            features = batch[0].to(device).unsqueeze(1)  # (batch_size, 1, input_size, input_size)
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

def inference(split, image, output):
    image_fmt = image.unsqueeze(0).to(device).unsqueeze(1)  # add batch and channel dimensions
    prediction = model(image_fmt).cpu()

    class_expected = output[0].item()
    class_predicted = prediction[:, :2].argmax(1).item()
    print(f"Expected class {class_expected} | Predicted class {class_predicted}")

    box_expected = output[1:] * input_size
    box_predicted = prediction[:, 2:].squeeze(0) * input_size
    print(f"Expected box {box_expected} | Predicted box {box_predicted}")

    image = image * 255.0
    image = image.type(torch.uint8)
    box_predicted = box_predicted.type(torch.int)
    box_expected = box_expected.type(torch.int)
    iou = metrics.intersection_over_union(box_predicted, box_expected).item()
    dataset.plot_image(image, class_predicted, box_predicted, box_expected, iou, split)


with torch.no_grad():
    for i in range(0, 10):
        print(f"Training Dataset {i}")
        image, output = train_dataset[i]
        inference("train", image, output)

    for i in range(0, 10):
        print(f"Dev Dataset {i}")
        image, output = dev_dataset[i]
        inference("dev", image, output)

    for i in range(0, 10):
        print(f"Validation Dataset {i}")
        image, output = val_dataset[i]
        inference("val", image, output)
