import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from torch.utils.data import Dataset, DataLoader, random_split

#############################
# DATA PREPROCESSING
#############################

raw_data = loadarff('datasets/Rice_MSC_Dataset.arff')
pd_df = pd.DataFrame(raw_data[0])

# Map DataFrame CLASS column string values to integer values
classe_names = sorted(list(pd_df['CLASS'].unique().astype(str)))
class_to_idx = {classe: idx for idx, classe in enumerate(classe_names)}
pd_df['CLASS'] = pd_df['CLASS'].map(lambda x: class_to_idx[x.decode("utf-8")])

# Normalize all other columns using by subtracting the mean and dividing by the standard deviation
for col in pd_df.columns:
    if col != 'CLASS':
        pd_df[col] = (pd_df[col] - pd_df[col].min()) / (pd_df[col].max() - pd_df[col].min())

pd_df = pd_df.dropna()
print("Number of records after removing NaN values: ", len(pd_df))


class RiceDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        features = record.drop('CLASS').values.astype(float)
        features = torch.tensor(features, dtype=torch.float32)
        label = record['CLASS'].astype(int)
        label = torch.tensor(label, dtype=torch.int8)
        return features, label


split = int(0.9 * len(pd_df))
rice_dataset = RiceDataset(pd_df)
train_dataset, test_dataset = random_split(rice_dataset, [split, len(pd_df) - split])

#############################
# MODEL DEFINITION
#############################
learning_rate = 1e-4
num_epochs = 10
batch_size = 128

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on device: {device}")


class ANN(torch.nn.Module):

    def __init__(self, num_features, num_classes, hidden_units=100):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=hidden_units)
        self.fc2 = torch.nn.Linear(in_features=hidden_units, out_features=num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = ANN(num_features=len(pd_df.columns) - 1, num_classes=len(classe_names))
model.to(device)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

#############################
# TRAINING
#############################

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(num_epochs):

    # Training
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    count = 0

    for batch in train_dataloader:
        features = batch[0].to(device)
        labels = batch[1].to(device)

        # Forward pass
        prediction = model(features)
        batch_loss = criterion(prediction, labels)
        batch_acc = (prediction.argmax(1) == labels).type(torch.float).sum().item()
        if count % 100 == 0:
            print(f"Batch {count} | Loss: {batch_loss:.4f} | Accuracy: {batch_acc / batch_size:.4f}")
        count += 1

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        epoch_acc += batch_acc

    print(f"Training Epoch {epoch} | "
          f"Loss: {epoch_loss / len(train_dataloader)} | "
          f"Accuracy: {epoch_acc / len(train_dataset)}")

    # Validation
    with torch.no_grad():
        model.eval()
        eval_loss = 0
        eval_acc = 0

        for batch in val_dataloader:

            features = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            prediction = model(features)
            batch_loss = criterion(prediction, labels)
            batch_acc = (prediction.argmax(1) == labels).type(torch.float).sum().item()

            eval_loss += batch_loss.item()
            eval_acc += batch_acc

        print(f"Validation | "
              f"Loss: {eval_loss / len(val_dataloader)} | "
              f"Accuracy: {eval_acc / len(test_dataset)}")
        print(80 * "-")

    history["train_loss"].append(epoch_loss / len(train_dataloader))
    history["train_acc"].append(epoch_acc / len(train_dataset))
    history["val_loss"].append(eval_loss / len(val_dataloader))
    history["val_acc"].append(eval_acc / len(test_dataset))

print("Training complete!")
print("Saving model...")
torch.save(model.state_dict(), 'output/rice_ann.pth')

# Plot history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot precision
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
