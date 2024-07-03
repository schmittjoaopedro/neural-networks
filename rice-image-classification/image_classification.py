import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader

from lenet import LeNet
from vgg16 import VGG16


def convert_image(entry):
    entry['image'] = torch.stack([T.ToTensor()(im) for im in entry['image']])
    entry['label'] = torch.tensor(entry['label'])
    return entry


if __name__ == '__main__':
    # ========================================
    # DATASET PROCESSING
    # ========================================

    # dataset = load_from_disk("datasets/rice.hf")
    # dataset = dataset.with_format("torch")

    dataset = load_dataset("nateraw/rice-image-dataset")['train']

    print("Dataset information")
    print("-- Columns", dataset.column_names)
    print("-- Shape", dataset.shape)
    print("-- Label", dataset.features['label'])
    print("-- Image", dataset.features['image'])

    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    train_dataset.set_transform(convert_image)
    val_dataset.set_transform(convert_image)
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
    EPOCHS = 1
    INPUT_DIM = image['image'].shape  # (num_channels, height, width) = (3, 256, 256)
    NUM_CLASSES = len(train_dataset.features['label'].names)  # 5 = (Arborio, Basmati, Ipsala, Jasmini, Karacadag)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=6, prefetch_factor=8, pin_memory=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=6, prefetch_factor=8, pin_memory=True)
    train_total_batches = len(train_dataset) // BATCH_SIZE
    val_total_batches = len(val_dataset) // BATCH_SIZE

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running on device: {device}")

    # Initializing
    model_name = "VGG16"
    if model_name == "LeNet":
        model = LeNet(
            num_channels=INPUT_DIM[0],
            image_width=INPUT_DIM[1],
            image_height=INPUT_DIM[2],
            num_classes=NUM_CLASSES
        ).to(device)
    else:
        model = VGG16(
            num_channels=INPUT_DIM[0],
            image_length=INPUT_DIM[1],
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

        for batch in train_data_loader:
            # Next batch of training samples
            # Convert batch list of tensor to a single tensor of size (batch_size, 3, 256, 256)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            prediction = model(images)
            batch_loss = criterion(prediction, labels)
            batch_acc = (prediction.argmax(1) == labels).type(torch.float).sum().item()
            counter += 1
            print(f"EPOCH: {epoch + 1}/{EPOCHS} | "
                  f"BATCH: {counter}/{train_total_batches} | "
                  f"BATCH LOSS: {batch_loss:.4f} | "
                  f"BATCH ACCURACY: {batch_acc / BATCH_SIZE:.4f}")

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Statistics
            total_train_loss += batch_loss
            total_train_correct += batch_acc

        with torch.no_grad():
            # Set the model in evaluation
            model.eval()
            counter = 0

            for batch in val_data_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                # Forward pass
                prediction = model(images)
                total_val_loss += criterion(prediction, labels)
                total_val_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
                counter += 1
                print(f"BATCH: {counter}/{val_total_batches}")

        average_train_loss = total_train_loss / len(train_dataset)
        average_train_acc = total_train_correct / len(train_dataset)
        average_val_loss = total_val_loss / len(val_dataset)
        average_val_acc = total_val_correct / len(val_dataset)
        history['train_loss'].append(average_train_loss)
        history['train_acc'].append(average_train_acc)
        history['val_loss'].append(average_val_loss)
        history['val_acc'].append(average_val_acc)
        print(f"EPOCH: {epoch + 1}/{EPOCHS} | "
              f"TRAIN LOSS: {average_train_loss:.4f} | "
              f"TRAIN ACCURACY: {average_train_acc:.4f} | "
              f"VALIDATION LOSS: {average_val_loss:.4f} | "
              f"VALIDATION ACCURACY: {average_val_acc:.4f}")

        torch.save(model.state_dict(), f"output/{model_name}_epoch_{epoch + 1}.pth")
