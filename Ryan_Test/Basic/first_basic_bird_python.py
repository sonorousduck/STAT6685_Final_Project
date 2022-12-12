import numpy as np
import librosa
import torch
import pandas as pd
import torchaudio
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score


class BirdClefDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration):
        self.audio_paths = df['filename'].values
        self.labels = df['primary_label_encoded'].values
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = f"data/{self.audio_paths[index]}"
        signal, sr = torchaudio.load(audio_path)

        # Check if our sample rate is the same as the target sameple rate. If not, resample
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        
        # Check shape and verify it is correct
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0, keepdim=True)
        
        # Check the number of samples and pad/truncate as needed
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        
        elif signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)
        
        # Then we can do signal processing. This tutorial uses the Mel Spectrogram, so I will leave that in for right now. This may not be what we want to go with in the end
        mel = self.transformation(signal)

        # Transforms mel into a 3 channel image (This is for RESNET)
        image = torch.cat([mel, mel, mel])

        # Normalize the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.tensor(self.labels[index])

        return image, label

df = pd.read_csv('data/train_metadata.csv')
df.head()


encoder = LabelEncoder()
df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])
df.head()

(X_train, X_test, y_train, y_test) = train_test_split(df, df['primary_label_encoded'], test_size= .2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=7)

sr = 32_000
n_fft = 1024
hop_length = 512
train_batch_size = 128
valid_batch_size = 128
num_classes = 152
duration = 7
n_mels = 64

def get_data():
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length,
                                                            n_mels=64)

    train_dataset = BirdClefDataset(X_train, mel_spectrogram, sr, duration)
    valid_dataset = BirdClefDataset(X_val, mel_spectrogram, sr, duration)

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, valid_batch_size, shuffle=False)

    return train_loader, valid_loader


# Simple CNN Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(224256, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_loader, valid_loader = get_data()

# Train Loop
load = True 
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
epochs = 150

if load:
    model.load_state_dict(torch.load('./model.bin'))

best_f1 = 0.0
total_f1 = []

for epoch in range(epochs):
    print(f"Starting epoch: {epoch}")
    loop = tqdm(train_loader, position=0)
    model.train()
    for i, (x, y) in enumerate(loop):
        y = y.type(torch.LongTensor)
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        # _, predictions = torch.max(outputs, 1)
        
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    # Run validation loop
    if True:
        model.eval()
        print("Checking validation score")
        loop_validation = tqdm(valid_loader, position=0)
        pred = []
        label = []

        for i, (X, y) in enumerate(loop_validation):
            y = y.type(torch.LongTensor)
            y = y.to(device)
            X = X.to(device)

            outputs = model(X)
            _, predictions = torch.max(outputs, 1)

            loss = criterion(outputs, y)

            pred.extend(predictions.view(-1).cpu().detach().numpy())
            label.extend(y.view(-1).cpu().detach().numpy())

            loop_validation.set_description(f"Validation Epoch [{epoch + 1}/{epochs}")
            loop_validation.set_postfix(loss=loss.item())

        valid_f1 = f1_score(label, pred, average='macro')
        total_f1.append(valid_f1)

        try:
            with open('f1_score.txt', 'a') as f:
                f.write(f"{valid_f1}\n")
        except:
            pass

        if valid_f1 > best_f1:
            print(f"Validation F1 Improved - {best_f1} ---> {valid_f1}")
            best_f1 = valid_f1
            torch.save(model.state_dict(), f'./model.bin')
            print(f"Saved model checkpoint at ./model.bin")
        else:
            print(f"Validation did NOT improve - {best_f1} <--- {valid_f1}")


