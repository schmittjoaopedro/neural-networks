import os.path

import torch
import torchvision.transforms as T
from datasets import load_dataset, load_from_disk

from lenet import LeNet

# ========================================
# DATASET PROCESSING
# ========================================
if not os.path.exists("datasets/rice.hf"):
    def convert_image(record):
        record['image'] = T.ToTensor()(record['image'].resize((100, 100)))
        return record


    dataset = load_dataset("nateraw/rice-image-dataset")['train']
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(convert_image)
    dataset.save_to_disk("datasets/rice.hf")

dataset = load_from_disk("datasets/rice.hf")
dataset = dataset.with_format("torch")
print("Dataset information")
print("-- Columns", dataset.column_names)
print("-- Shape", dataset.shape)
print("-- Label", dataset.features['label'])
print("-- Image", dataset.features['image'])

dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
val_dataset = dataset['test']
print("Train dataset", train_dataset.shape)
print("Val dataset", val_dataset.shape)

idx = 2
image = train_dataset[idx]
print("First image")
print("-- image label", image['label'])
print("-- image shape", image['image'].shape)

# plt.imshow(image['image'])
# plt.title(train_dataset.features['label'].int2str(image['label']))
# plt.show()


# ========================================
# MODEL TRAINING
# ========================================
# Hyperparameters
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
INPUT_DIM = image['image'].shape  # (num_channels, height, width) = (3, 256, 256)
NUM_CLASSES = len(train_dataset.features['label'].names)  # 5 = (Arborio, Basmati, Ipsala, Jasmini, Karacadag)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Initializing
model = LeNet(
    num_channels=INPUT_DIM[0],
    image_width=INPUT_DIM[1],
    image_height=INPUT_DIM[2],
    num_classes=NUM_CLASSES
).to(device)
criterion = torch.nn.NLLLoss()  # Negative Log Likelihood Loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# Training
for epoch in range(0, EPOCHS):
    # Set the model in training mode
    model.train()

    # Loss function cost and correct predictions
    total_train_loss = 0
    total_train_correct = 0
    total_val_loss = 0
    total_val_correct = 0
    counter = 0

    for batch in train_dataset.iter(batch_size=BATCH_SIZE):
        # Next batch of training samples
        # Convert batch list of tensor to a single tensor of size (batch_size, 3, 256, 256)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        prediction = model(images)
        loss = criterion(prediction, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_train_loss += loss
        total_train_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
        counter += 1
        print(f"EPOCH: {epoch + 1}/{EPOCHS} | BATCH {counter}/{len(train_dataset) // BATCH_SIZE}", end="\r")

    with torch.no_grad():
        # Set the model in evaluation
        model.eval()

        for batch in val_dataset.iter(batch_size=BATCH_SIZE):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            prediction = model(images)
            total_val_loss += criterion(prediction, labels)
            total_val_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()

    average_train_loss = total_train_loss / len(train_dataset)
    average_train_acc = total_train_correct / len(train_dataset)
    average_val_loss = total_val_loss / len(val_dataset)
    average_val_acc = total_val_correct / len(val_dataset)
    history['train_loss'].append(average_train_loss)
    history['train_acc'].append(average_train_acc)
    history['val_loss'].append(average_val_loss)
    history['val_acc'].append(average_val_acc)
    print(f"EPOCH: {epoch + 1}/{EPOCHS} | "
          f"Train Loss: {average_train_loss:.4f} | Train Acc: {average_train_acc:.4f} | "
          f"Val Loss: {average_val_loss:.4f} | Val Acc: {average_val_acc:.4f}")
