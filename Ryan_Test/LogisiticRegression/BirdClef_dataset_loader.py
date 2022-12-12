import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import cv2

class BirdClefDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration, is_train, is_3d=True):
        self.audio_paths = df['filename'].values
        self.labels = df['primary_label_encoded'].values
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration
        self.is_train = is_train
        self.is_3d = is_3d # Should be true to train ResNet and so forth
    
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
        
        # Then we can do signal processing. This tutorial uses the Mel Spectrogram, so I will leave that in for right now. This may not be what we want to go with in the end
        mel = self.transformation(signal)


        if self.is_train:
            mel = torch.tensor(self.test_image_augmentation(mel))
        image = torch.cat([mel, mel, mel]) if self.is_3d else mel

        # Normalize the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.tensor(self.labels[index])

        return image, label