import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BirdClefDataset(Dataset):
    def __init__(self, df, target_sample_rate, duration):
        self.audio_paths = df['filename'].values
        self.labels = df['primary_label_encoded'].values
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration


        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                n_fft=n_fft,
                                                                win_length=25,
                                                                hop_length=10,
                                                                n_mels=128)
        self.mel_spectrogram_1 = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                n_fft=n_fft,
                                                                win_length=50,
                                                                hop_length=25,
                                                                n_mels=128)
        self.mel_spectrogram_2 = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                n_fft=n_fft,
                                                                win_length=100,
                                                                hop_length=50,
                                                                n_mels=128)    
    
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
        
        #  Implementing idea from "Rethinking CNN Models for Audio Classification"
        #  The idea here is to create 3-channels by doing melspectrograms of different window sizes and hop lengths
        #  Originally, I just did mel concatenated 3 times, which got up to 57% accuracy
                                                    

        # Then we can do signal processing. This tutorial uses the Mel Spectrogram, so I will leave that in for right now. This may not be what we want to go with in the end
        mel = self.mel_spectrogram(signal)
        mel_1 = self.mel_spectrogram_1(signal)
        mel_2 = self.mel_spectrogram_2(signal)

        print(len(mel[0][0]))
        print(len(mel_1[0][0]))
        print(len(mel_2[0][0]))

        # Transforms mel into a 3 channel image (This is for RESNET)
        image = torch.cat([mel, mel_1, mel_2])

        # Normalize the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.tensor(self.labels[index])

        return image, label

def get_data(fold):

    train_df = df[df['fold'] != fold].reset_index(drop = True)
    valid_df = df[df['fold'] == fold].reset_index(drop = True)

    train_dataset = BirdClefDataset(train_df, sr, duration)
    valid_dataset = BirdClefDataset(valid_df, sr, duration)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=24)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=24)
    
    return train_loader, valid_loader

def run(fold):
  train_loader, valid_loader = get_data(fold)

  load = False
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(resnet50.parameters(), lr=1e-4)
  epochs = 200
  train_epoch_losses = []
  valid_epoch_losses = []
  valid_epoch_accuracy = []
  early_stop = 5

  if load:
    resnet50.load_state_dict(torch.load('./resnet50.bin'))


  for epoch in range(epochs):
    loop = tqdm(train_loader)
    resnet50.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(loop):
      y = y.type(torch.LongTensor)
      y = y.to(device)
      x = x.to(device)

      outputs = resnet50(x)

      _, predictions = torch.max(outputs, 1)
      loss = criterion(outputs, y)
      loss.backward()
      epoch_loss += loss.item()
      optimizer.step()
      optimizer.zero_grad()

      loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
      loop.set_postfix(loss=(epoch_loss / (i + 1)))
    train_epoch_losses.append(epoch_loss / len(train_loader))

    # Validation
    loop_validation = tqdm(valid_loader)
    resnet50.eval()
    temp_loss = 0
    temp_accuracy = 0
    for i, (x, y) in enumerate(loop_validation):
      y = y.type(torch.LongTensor)
      y = y.to(device)
      x = x.to(device)

      outputs = resnet50(x)
      _, predictions = torch.max(outputs, 1)
      loss = criterion(outputs, y)
      temp_loss += loss.item()
      accuracy = ((predictions.detach().cpu().numpy() == y.detach().cpu().numpy()).mean())
      temp_accuracy += accuracy


      loop_validation.set_description(f"Validation Epoch [{epoch + 1}/{epochs}")
      loop_validation.set_postfix_str(f"Loss: {round(temp_loss / (i + 1), 3)} Accuracy: {round(temp_accuracy / (i + 1), 3)}")
    valid_epoch_losses.append(temp_loss / len(valid_loader))
    valid_epoch_accuracy.append(temp_accuracy / len(valid_loader))
    early_stop -= 1
    # Early Stopping Criteria
    if early_stop <= 0:
      if valid_epoch_accuracy[-1] < valid_epoch_accuracy[-5]:
        print(f"Early stopping because most recent accuracy {valid_epoch_accuracy[-1]}, 5 ago was {valid_epoch_accuracy[-5]}")
        break
      else:
        early_stop = 5

  torch.save(resnet50.state_dict(), f'./resnet50_unfrozen_different_mel.bin')
  return train_epoch_losses, valid_epoch_losses, valid_epoch_accuracy

  


if __name__ == "__main__":
  df = pd.read_csv('data/train_metadata.csv')
  encoder = LabelEncoder()
  df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])

  skf = StratifiedKFold(n_splits=5)
  for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label_encoded'])):
      df.loc[val_ind, 'fold'] = k


  sr = 32_000
  n_fft = 1024
  hop_length = 512
  train_batch_size = 32
  valid_batch_size = 64
  num_classes = 152
  duration = 7
  n_mels = 64

  resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
  utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')


  # Accuracy for frozen layers only hit 27.5%. Attempting to leave all layers unfrozen
  # Freeze all layers
  # for paramt in resnet50.parameters():
  #     paramt.requires_grad = False

  # Change the final layer to fit our purposes
  resnet50.fc = nn.Sequential(
    nn.Linear(2048, 1024), 
            nn.ReLU(), 
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(p=0.2),
            nn.Linear(512, 152)
  )

  # resnet50.fc = nn.Linear(in_features=2048, out_features=152, bias=True)
  resnet50.to(device)



fold = 2
train_epoch_losses, valid_epoch_losses, valid_epoch_accuracy = run(fold)

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
plt.suptitle("All Frozen Layers, 1 Trainable Output Layer")
ax[0].plot(train_epoch_losses, label="Train Loss")
ax[0].plot(valid_epoch_losses, label="Validation Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss over Epochs")
ax[0].legend()

ax[1].plot(valid_epoch_accuracy, label="Validation Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy over Epochs")

plt.savefig('./loss_accuracy_unfrozen_deeper_different.jpg')