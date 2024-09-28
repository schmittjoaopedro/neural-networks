import gzip
import os
import shutil
import time

import pandas as pd
import requests
import torch
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast

# Configure file logger
import logging

logging.basicConfig(
    filename='fine-tunning-bert.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

torch.manual_seed(123)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.log(logging.INFO, f"Using device: {device}")

NUM_EPOCHS = 3

# Check if movie_data.csv exists, if not download it

if not os.path.isfile("movie_data.csv"):
    url = "https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz"
    filename = url.split("/")[-1]

    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    with gzip.open("movie_data.csv.gz", "rb") as f_in:
        with open("movie_data.csv", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

df = pd.read_csv('movie_data.csv')
logging.log(logging.INFO, df.head(3))

train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values

valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values

test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset = IMDbDataset(valid_encodings, valid_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            ### Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()

            if batch_idx % 10 == 0:
                logging.log(logging.INFO, f'Batch {batch_idx:03d}/{len(data_loader):03d}')

    return correct_pred.float() / num_examples * 100


# Fine-tuning
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch_idx, batch in enumerate(train_loader):

        ### Prepare data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        ### Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']

        ### Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        ### Logging
        if not batch_idx % 10:
            logging.log(logging.INFO, f'Epoch: {epoch + 1:04d}/{NUM_EPOCHS:04d}'
                                      f' | Batch {batch_idx:04d}/{len(train_loader):04d}'
                                      f' | Loss: {loss:.4f}')

    model.eval()

    with torch.set_grad_enabled(False):
        logging.log(logging.INFO, f"Train acc: {compute_accuracy(model, train_loader, device):.2f}")
        logging.log(logging.INFO, f"Valid acc: {compute_accuracy(model, valid_loader, device):.2f}")

    logging.log(logging.INFO, f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

logging.log(logging.INFO, f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
logging.log(logging.INFO, f'Test accuracy: {compute_accuracy(model, test_loader, device):.2f}%')
