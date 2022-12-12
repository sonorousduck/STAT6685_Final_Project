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
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


class BirdClefDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration, is_train):
        self.audio_paths = df['filename'].values
        self.labels = df['primary_label_encoded'].values
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration
        self.is_train = is_train 
      
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, index):
        audio_path = f"../data/{self.audio_paths[index]}"
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
        
        mel = self.transformation(signal)

        # NO AUGMENTATION ON THIS FILE
        image = torch.cat([mel, mel, mel])
        

        # Transforms mel into a 3 channel image (This is for RESNET)
        # Normalize the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.tensor(self.labels[index])

        return image, label

def get_data(fold):
  mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                          n_fft=n_fft,
                                                          hop_length=hop_length,
                                                          n_mels=64)
  train_df = df[df['fold'] != fold].reset_index(drop = True)
  valid_df = df[df['fold'] == fold].reset_index(drop = True)


  train_df, test_df = train_test_split(train_df, test_size=0.1)

  train_dataset = BirdClefDataset(train_df, mel_spectrogram, sr, duration, True)
  valid_dataset = BirdClefDataset(valid_df, mel_spectrogram, sr, duration, False)
  test_dataset = BirdClefDataset(test_df, mel_spectrogram, sr, duration, False)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=24)
  valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=24)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=24)
  
  return train_loader, valid_loader, test_loader

def run(fold):
  train_loader, valid_loader, test_loader = get_data(fold)

  load = False
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(regnet.parameters(), lr=1e-4)
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
  epochs = 1000
  train_epoch_losses = []
  valid_epoch_losses = []
  valid_epoch_accuracy = []
  valid_epoch_f1 = []

  early_stop = 10
  last_accuracy = 0

  if load:
    regnet.load_state_dict(torch.load('./regnet_no_augmentation_class_imbalance.bin'))


  for epoch in range(epochs):
    loop = tqdm(train_loader)
    regnet.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(loop):
      y = y.type(torch.LongTensor)
      y = y.to(device)
      x = x.to(device)

      outputs = regnet(x)
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
    regnet.eval()
    temp_loss = 0
    pred = []
    labels = []
    for i, (x, y) in enumerate(loop_validation):
      y = y.type(torch.LongTensor)
      y = y.to(device)
      x = x.to(device)

      outputs = regnet(x)
      _, predictions = torch.max(outputs, 1)
      loss = criterion(outputs, y)
      temp_loss += loss.item()
      pred.extend(predictions.view(-1).cpu().detach().numpy())
      labels.extend(y.view(-1).cpu().detach().numpy())
      accuracy = accuracy_score(labels, pred)
      f1 = f1_score(labels, pred, average='macro')

      loop_validation.set_description(f"Validation Epoch [{epoch + 1}/{epochs}")
      loop_validation.set_postfix_str(f"Loss: {round(temp_loss / (i + 1), 3)} Accuracy: {round(accuracy, 3)} and F1 score: {round(f1, 3)}")
    
    valid_accuracy = accuracy_score(labels, pred)
    valid_f1 = f1_score(labels, pred, average='macro')
    valid_epoch_f1.append(valid_f1)
    valid_epoch_losses.append(temp_loss / len(valid_loader))
    valid_epoch_accuracy.append(valid_accuracy)
    scheduler.step(temp_loss / len(valid_loader))
    
    # Early Stopping Criteria
    if (last_accuracy < valid_accuracy):
        print(f"Accuracy improved from {last_accuracy} -> {valid_accuracy}. Saving Model")
        last_accuracy = valid_accuracy
        torch.save(regnet.state_dict(), f'./regnet_no_augmentation_class_imbalance.bin')
        early_stop = 10
    else:
      early_stop -= 1


    if early_stop <= 0:
      print(f"Early stopping because accuracy hasn't improved for the last 5 epochs")
      break
    

  # Test
  regnet.eval()
  loop_test = tqdm(test_loader)
  temp_loss = 0
  pred = []
  labels = []
  for i, (x, y) in enumerate(loop_test):
    y = y.type(torch.LongTensor)
    y = y.to(device)
    x = x.to(device)

    outputs = regnet(x)
    _, predictions = torch.max(outputs, 1)
    loss = criterion(outputs, y)
    temp_loss += loss.item()
    pred.extend(predictions.view(-1).cpu().detach().numpy())
    labels.extend(y.view(-1).cpu().detach().numpy())

    accuracy = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='macro')


    loop_test.set_description(f"Test Epoch")
    loop_test.set_postfix_str(f"Loss: {round(temp_loss / (i + 1), 3)} Accuracy: {round(accuracy, 3)}, F1: {round(f1, 3)}")

  test_loss = temp_loss / len(test_loader)
  test_accuracy = accuracy_score(labels, pred)
  test_f1 = f1_score(labels, pred, average='macro')

  return train_epoch_losses, valid_epoch_losses, valid_epoch_accuracy, valid_epoch_f1, test_loss, test_accuracy, test_f1

  


if __name__ == "__main__":
  df = pd.read_csv('../data/train_metadata.csv')

  encoder = LabelEncoder()
  df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])

  skf = StratifiedKFold(n_splits=5)
  for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label_encoded'])):
      df.loc[val_ind, 'fold'] = k
  
  v = df.primary_label.value_counts()
  df = df[df.primary_label.isin(v.index[v.gt(50)])]

  sr = 32_000
  n_fft = 1024
  hop_length = 512
  train_batch_size = 32
  valid_batch_size = 64
  num_classes = 152
  duration = 7
  n_mels = 64




fold = 1

for i in range(fold):

  
  regnet = torchvision.models.regnet_x_32gf(pretrained=True)

  regnet.fc = nn.Linear(in_features=2520, out_features=152) # 75% accuracy on this version
  regnet.to(device)

  train_epoch_losses, valid_epoch_losses, valid_epoch_accuracy, valid_epoch_f1, test_loss, test_accuracy, test_f1 = run(4)
  print(f"Test Loss: {test_loss} with an accuracy of {test_accuracy} and f1: {test_f1}")
  fig, ax = plt.subplots(1, 3, figsize=(10, 7))
  plt.suptitle(f"RegNet No Augmentation Less Class Imbalance")
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
  ax[1].legend()

  ax[2].plot(valid_epoch_f1, label="Validation F1")
  ax[2].set_xlabel("Epochs")
  ax[2].set_ylabel("F1 Score")
  ax[2].set_title("F1 Score over Epochs")
  ax[2].legend()

  plt.savefig(f'./regnet_no_augmentation_class_imbalance.jpg')

  train_epoch_losses = np.array(train_epoch_losses)
  valid_epoch_losses = np.array(valid_epoch_losses)
  valid_epoch_accuracy = np.array(valid_epoch_accuracy)
  valid_epoch_f1 = np.array(valid_epoch_f1)

  np.savetxt('train_epoch_losses_regnet_no_augmentation_class_imbalance.txt', train_epoch_losses, delimiter=',')
  np.savetxt('valid_epoch_losses_regnet_no_augmentation_class_imbalance.txt', valid_epoch_losses, delimiter=',')
  np.savetxt('valid_epoch_accuracy_regnet_no_augmentation_class_imbalance.txt', valid_epoch_accuracy, delimiter=',')
  np.savetxt('valid_epoch_f1_regnet_no_augmentation_class_imbalance.txt', valid_epoch_f1, delimiter=',')
  np.savetxt('test_loss_accuracy_f1_regnet_no_augmentation_class_imbalance.txt', np.array([test_loss, test_accuracy, test_f1]))
