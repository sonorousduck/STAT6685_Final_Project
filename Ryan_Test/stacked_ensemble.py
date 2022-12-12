# https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import torchvision
from sklearn.metrics import accuracy_score
from numpy import dstack
import numpy as np
import BirdClef_dataset_loader
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
import cv2
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sr = 32_000
n_fft = 1024
hop_length = 512
train_batch_size = 32
valid_batch_size = 32
num_classes = 152
duration = 7
n_mels = 64


def load_all_models():
    all_models = []
    
    regnet = torchvision.models.regnet_x_32gf(pretrained=False)
    regnet.fc = nn.Linear(in_features=2520, out_features=152) # 75% accuracy on this version
    regnet.load_state_dict(torch.load('regnet.bin'))
    regnet.to(device)


    vgg19 = torchvision.models.vgg19_bn(pretrained=False)
    vgg19.fc = nn.Linear(in_features=4096, out_features=152)
    vgg19.load_state_dict(torch.load('./vgg19.bin'))
    vgg19.to(device)

    resnet152 = torchvision.models.resnet152(pretrained=False)
    resnet152.fc = nn.Linear(in_features=2048, out_features=152)
    resnet152.load_state_dict(torch.load('resnet152.bin'))
    resnet152.to(device)



    all_models.append(regnet)
    # all_models.append(vgg19)
    all_models.append(resnet152)

    return all_models

# This becomes the dataset to input into the classification model after the RegNet, VGG19, and ResNet152
def stacked_dataset(models, x):
  stackX = None
  for model in models:
      yhat = model(x)
      if stackX is None:
          stackX = yhat.detach().cpu().numpy()
      else:
          stackX = np.dstack((stackX, yhat.detach().cpu().numpy()))
  stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

  return stackX

def train_stacked_model(models, model, X, y):
    stackedX = stacked_dataset(models, X)
    model.fit(stackedX, y.detach().cpu().numpy())


def get_data(fold=4):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=64)
    train_df = df[df['fold'] != fold].reset_index(drop = True)
    valid_df = df[df['fold'] == fold].reset_index(drop = True)

    train_dataset = BirdClef_dataset_loader.BirdClefDataset(train_df, mel_spectrogram, sr, duration, True)
    valid_dataset = BirdClef_dataset_loader.BirdClefDataset(valid_df, mel_spectrogram, sr, duration, False)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=24)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=24)
    
    return train_loader, valid_loader

def stacked_prediction(models, model, input_x):
    stacked_x = stacked_dataset(models, input_x)
    yhat = model.predict(stacked_x)
    return yhat



if __name__ == "__main__":
    epochs = 30
    df = pd.read_csv('data/train_metadata.csv')
    encoder = LabelEncoder()
    df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])

    model = LogisticRegression()
    all_models = load_all_models()
    skf = StratifiedKFold(n_splits=5)
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label_encoded'])):
        df.loc[val_ind, 'fold'] = k
    
    train_loader, valid_loader = get_data(4)
    for epoch in range(epochs):
        loop = tqdm(train_loader)
        
        for i, (x, y) in enumerate(loop):
          y = y.type(torch.LongTensor)
          y = y.to(device)
          x = x.to(device)

          # for individual_model in all_models:
          #     yhat = individual_model(x)
          #     accuracy = accuracy_score(y.detach().cpu().numpy(), torch.argmax(yhat.detach(), axis=1).cpu().numpy())
          #     print(accuracy)
      
          # break
          train_stacked_model(all_models, model, x, y)
          yhat = stacked_prediction(all_models, model, x)
          acc = accuracy_score(y.detach().cpu().numpy(), yhat)
          loop.set_postfix_str('Stacked Train Accuracy: %.3f' % acc)

        validation_loop = tqdm(valid_loader)
        for i, (x, y) in enumerate(validation_loop):
          y = y.type(torch.LongTensor)
          y = y.to(device)
          x = x.to(device)


          yhat = stacked_prediction(all_models, model, x)
          acc = accuracy_score(y.detach().cpu().numpy(), yhat)
          validation_loop.set_postfix_str('Stacked Test Accuracy: %.3f' % acc)


        


            
            

    


    

    
