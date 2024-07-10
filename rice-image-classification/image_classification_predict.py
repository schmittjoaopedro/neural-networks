import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets import load_dataset
from lenet import LeNet
from vgg16 import VGG16

model_name = "VGG16"
batch_size = 64


def convert_image(entry):
    entry['image'] = torch.stack([T.ToTensor()(im) for im in entry['image']])
    entry['label'] = torch.tensor(entry['label'])
    return entry


if __name__ == '__main__':
    dataset = load_dataset("nateraw/rice-image-dataset")['train']
    dataset = dataset.train_test_split(test_size=0.1)

    val_dataset = dataset['test']
    val_dataset.set_transform(convert_image)
    print("Val dataset", val_dataset.shape)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on device: {device}")

    image = val_dataset[0]
    INPUT_DIM = image['image'].shape  # (num_channels, height, width) = (3, 256, 256)
    NUM_CLASSES = len(val_dataset.features['label'].names)  # 5 = (Arborio, Basmati, Ipsala, Jasmini, Karacadag)
    print("Input dimension", INPUT_DIM)
    print("Number of classes", NUM_CLASSES)

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
            num_classes=NUM_CLASSES,
            dropout=0.5,
        ).to(device)

    model.load_state_dict(torch.load(f"output/{model_name}_epoch_1.pth"))
    
    print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=6, prefetch_factor=8, pin_memory=True)
    val_total_batches = len(val_dataset) // batch_size

    counter = 0
    total_val_loss = 0
    total_val_correct = 0
    criterion = torch.nn.NLLLoss()

    with torch.no_grad():
        model.eval()

        for batch in val_data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            prediction = model(images)
            total_val_loss += criterion(prediction, labels)
            total_val_correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
            counter += 1
            print(f"BATCH: {counter}/{val_total_batches}")

    average_val_loss = total_val_loss / len(val_dataset)
    average_val_acc = total_val_correct / len(val_dataset)
    history['val_loss'].append(average_val_loss)
    history['val_acc'].append(average_val_acc)
    print(f"VALIDATION LOSS: {average_val_loss:.4f} | "
          f"VALIDATION ACCURACY: {average_val_acc:.4f}")
