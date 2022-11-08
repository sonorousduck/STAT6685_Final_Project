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
import cv2
from torchvision.models import efficientnet_v2_l
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BirdClefDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration, is_train):
        self.audio_paths = df['filename'].values
        self.labels = df['primary_label_encoded'].values
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration
        self.is_train = is_train
    
    def test_image_augmentation(self, img):
      IM_AUGMENTATION = {#'type':[probability, value]
                   'roll':[0.5, (0.0, 0.05)], 
                   'noise':[0.1, 0.01],
                   'noise_samples':[0.1, 1.0],
                   'brightness':[0.5, (0.25, 1.25)],
                  #  'crop':[0.5, 0.07],
                  #  'flip': [0.25, 1]
                   }
      RANDOM_SEED = 7
      RANDOM = np.random.RandomState(RANDOM_SEED)
      IM_SIZE = (512, 256)

      if 'crop' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['crop'][0], 1 - IM_AUGMENTATION['crop'][0]]):
          h, w = img.shape[:2]
          cropw = RANDOM.randint(1, int(float(w) * IM_AUGMENTATION['crop'][1]))
          croph = RANDOM.randint(1, int(float(h) * IM_AUGMENTATION['crop'][1]))
          img = img[croph:-croph, cropw:-cropw]
          img = cv2.resize(img, (IM_SIZE[0], IM_SIZE[1]))

      #Flip - 1 = Horizontal, 0 = Vertical
      if 'flip' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['flip'][0], 1 - IM_AUGMENTATION['flip'][0]]):    
          img = cv2.flip(img, IM_AUGMENTATION['flip'][1])

      #Wrap shift (roll up/down and left/right)
      if 'roll' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['roll'][0], 1 - IM_AUGMENTATION['roll'][0]]):
          img = np.roll(img, int(img.shape[0] * (RANDOM.uniform(-IM_AUGMENTATION['roll'][1][1], IM_AUGMENTATION['roll'][1][1]))), axis=0)
          img = np.roll(img, int(img.shape[1] * (RANDOM.uniform(-IM_AUGMENTATION['roll'][1][0], IM_AUGMENTATION['roll'][1][0]))), axis=1)

      #substract/add mean
      if 'mean' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['mean'][0], 1 - IM_AUGMENTATION['mean'][0]]):   
          img += np.mean(img) * IM_AUGMENTATION['mean'][1]

      #gaussian noise
      if 'noise' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['noise'][0], 1 - IM_AUGMENTATION['noise'][0]]):
          img += RANDOM.normal(0.0, RANDOM.uniform(0, IM_AUGMENTATION['noise'][1]**0.5), img.shape)
          img = np.clip(img, 0.0, 1.0)

      #add noise samples
      # if 'noise_samples' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['noise_samples'][0], 1 - IM_AUGMENTATION['noise_samples'][0]]):
      #     img += openImage(NOISE[RANDOM.choice(range(0, len(NOISE)))], True) * IM_AUGMENTATION['noise_samples'][1]
      #     img -= img.min(axis=None)
      #     img /= img.max(axis=None)

      #adjust brightness
      if 'brightness' in IM_AUGMENTATION and RANDOM.choice([True, False], p=[IM_AUGMENTATION['brightness'][0], 1 - IM_AUGMENTATION['brightness'][0]]):
          img *= RANDOM.uniform(IM_AUGMENTATION['brightness'][1][0], IM_AUGMENTATION['brightness'][1][1])
          img = np.clip(img, 0.0, 1.0)

      return img

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

        # preprocess = transforms.Compose([
        #     transforms.Resize(299),
        #     transforms.CenterCrop(299),
        #     transforms.Normalize(mean=[0.485], std=[0.229]),
        # ])
        # mel = preprocess(mel)
        if self.is_train:
            mel = torch.tensor(self.test_image_augmentation(mel))
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

    train_dataset = BirdClefDataset(train_df, mel_spectrogram, sr, duration, True)
    valid_dataset = BirdClefDataset(valid_df, mel_spectrogram, sr, duration, False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=24)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=24)
    
    return train_loader, valid_loader

def run(fold):
  train_loader, valid_loader = get_data(fold)

  load = False
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(efficient_net.parameters(), lr=1e-4)
  scheduler = ReduceLROnPlateau(optimizer, 'min')
  epochs = 1000
  train_epoch_losses = []
  valid_epoch_losses = []
  valid_epoch_accuracy = []
  early_stop = 10
  last_accuracy = 0

  if load:
    efficient_net.load_state_dict(torch.load('./efficient_net.bin'))


  for epoch in range(epochs):
    loop = tqdm(train_loader)
    efficient_net.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(loop):
      y = y.type(torch.LongTensor)
      y = y.to(device)
      x = x.to(device)

      outputs = efficient_net(x)
      loss = criterion(outputs, y)
    #   loss2 = criterion(aux_outputs, y)
    #   loss = loss1 + 0.4 * loss2
    #   predictions = torch.max(outputs, 1)
      

    #   loss = criterion(outputs, y)
      loss.backward()
      epoch_loss += loss.item()
      optimizer.step()
      optimizer.zero_grad()

      loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
      loop.set_postfix(loss=(epoch_loss / (i + 1)))
    train_epoch_losses.append(epoch_loss / len(train_loader))

    # Validation
    loop_validation = tqdm(valid_loader)
    efficient_net.eval()
    temp_loss = 0
    temp_accuracy = 0
    for i, (x, y) in enumerate(loop_validation):
      y = y.type(torch.LongTensor)
      y = y.to(device)
      x = x.to(device)

      outputs = efficient_net(x)
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
    scheduler.step(temp_loss / len(valid_loader))
    
    # Early Stopping Criteria
    if (last_accuracy < temp_accuracy / len(valid_loader)):
        print(f"Accuracy improved from {last_accuracy} -> {temp_accuracy / len(valid_loader)}. Saving Model")
        last_accuracy = temp_accuracy / len(valid_loader)
        torch.save(efficient_net.state_dict(), f'./efficient_net.bin')


    if early_stop <= 0:
      if valid_epoch_accuracy[-1] < valid_epoch_accuracy[-10]:
        print(f"Early stopping because most recent accuracy {valid_epoch_accuracy[-1]}, 5 ago was {valid_epoch_accuracy[-10]}")
        break
      else:
        early_stop = 10
    
# Don't save here... We only want to save if it was the best performing network.
#   torch.save(regnet.state_dict(), f'./regnet.bin')
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




fold = 1

for i in range(fold):

  
  efficient_net = efficientnet_v2_l(pretrained=True)

  # Accuracy for frozen layers only hit 27.5%. Attempting to leave all layers unfrozen
  # Freeze all layers
  # for paramt in resnet50.parameters():
  #     paramt.requires_grad = False

  # Change the final layer to fit our purposes
#   inception.fc = nn.Sequential(
#     nn.Linear(2048, 1024), 
#             nn.ReLU(), 
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, 512), 
#             nn.ReLU(), 
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 256), 
#             nn.ReLU(), 
#             nn.Dropout(p=0.5),
#             nn.Linear(256, 152)
#   )

  efficient_net.fc = nn.Linear(in_features=1280, out_features=152)
  efficient_net.to(device)

  train_epoch_losses, valid_epoch_losses, valid_epoch_accuracy = run(4)

  fig, ax = plt.subplots(1, 2, figsize=(10, 7))
  plt.suptitle(f"EfficientNet")
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

  plt.savefig(f'./efficient_net.jpg')