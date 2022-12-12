import sklearn
from sklearn.linear_model import LogisticRegression
import BirdClef_dataset_loader
import torchaudio
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


sr = 32_000
n_fft = 1024
hop_length = 512
train_batch_size = 32
valid_batch_size = 32
num_classes = 152
duration = 7
n_mels = 64


class linear_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28_032, 152)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        out = torch.sigmoid(self.linear(x))
        return out


def get_data(is_3d=True):
    df = pd.read_csv('../data/train_metadata.csv')
    encoder = LabelEncoder()
    df['primary_label_encoded'] = encoder.fit_transform(df['primary_label'])
    fold = 4  # For now. TODO: Need to make it for loop over fold
    skf = StratifiedKFold(n_splits=5)
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['primary_label_encoded'])):
        df.loc[val_ind, 'fold'] = k

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                           n_fft=n_fft,
                                                           hop_length=hop_length,
                                                           n_mels=64)
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    train_df, test_df = train_test_split(train_df, test_size=0.1)

    train_dataset = BirdClef_dataset_loader.BirdClefDataset(
        train_df, mel_spectrogram, sr, duration, True, is_3d)
    valid_dataset = BirdClef_dataset_loader.BirdClefDataset(
        valid_df, mel_spectrogram, sr, duration, False, is_3d)
    test_dataset = BirdClef_dataset_loader.BirdClefDataset(
        test_df, mel_spectrogram, sr, duration, False, is_3d)

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=24)
    valid_loader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, shuffle=True, num_workers=24)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=24)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    epochs = 200
    early_stop = 10
    best_accuracy = 0
    model = linear_regression().cuda()

    train_loader, valid_loader, test_loader = get_data(is_3d=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    valid_epoch_f1 = []

    for epoch in range(epochs):
        train_loop = tqdm(train_loader)
        train_loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loop):
            y = y.type(torch.int64)
            y = y.cuda()
            x = x.cuda()
            y_pred = model(x)

            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            # train_accuracy = accuracy_score(y, y_pred)
            train_loss += loss.item()
            train_loop.set_postfix_str(f"Train loss: {train_loss / (i + 1)}")
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        valid_loop = tqdm(valid_loader)
        model.eval()
        valid_accuracy = 0
        valid_loss = 0
        pred = []
        labels = []
        for i, (x, y) in enumerate(valid_loop):
            y = y.type(torch.int64)
            y = y.cuda()
            x = x.cuda()
            y_pred = model(x)
            _, predictions = torch.max(y_pred, 1)

            loss = criterion(y_pred, y)
            pred.extend(predictions.view(-1).cpu().detach().numpy())
            labels.extend(y.view(-1).cpu().detach().numpy())
            accuracy = accuracy_score(labels, pred)
            f1 = f1_score(labels, pred, average='macro')

            valid_loss += loss.item()
            valid_loop.set_postfix_str(
                f"Accuracy: {round(accuracy, 4)} and valid loss: {round(valid_loss / (i + 1), 3)} and f1: {round(f1, 3)}")

        valid_loss /= len(valid_loop)
        valid_accuracy = accuracy_score(labels, pred)
        valid_f1 = f1_score(labels, pred, average='macro')
        valid_epoch_f1.append(valid_f1)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        if (valid_accuracy > best_accuracy):
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), f'./logistic_regression.bin')
            early_stop = 10
        else:
            early_stop -= 1

        if early_stop < 0:
            print("Early stopping, no increase in accuracy over the past 5 epochs")
            break
        scheduler.step(valid_loss)

    model.eval()
    loop_test = tqdm(test_loader)
    temp_loss = 0
    pred = []
    labels = []
    for i, (x, y) in enumerate(loop_test):
        y = y.type(torch.LongTensor)
        y = y.cuda()
        x = x.cuda()

        outputs = model(x)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        temp_loss += loss.item()
        pred.extend(predictions.view(-1).cpu().detach().numpy())
        labels.extend(y.view(-1).cpu().detach().numpy())

        accuracy = accuracy_score(labels, pred)
        f1 = f1_score(labels, pred, average='macro')

        loop_test.set_description(f"Test Epoch")
        loop_test.set_postfix_str(
            f"Loss: {round(temp_loss / (i + 1), 3)} Accuracy: {round(accuracy, 3)}, F1: {round(f1, 3)}")

    test_loss = temp_loss / len(test_loader)
    test_accuracy = accuracy_score(labels, pred)
    test_f1 = f1_score(labels, pred, average='macro')
    print(
        f"Test Loss: {test_loss} with an accuracy of {test_accuracy} and f1: {test_f1}")

    fix, ax = plt.subplots(1, 3, figsize=(10, 7))
    plt.suptitle(f"Logistic Regression")

    ax[0].set_title("Validation Accuracy")
    ax[0].plot(valid_accuracies)
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].set_title("Losses")
    ax[1].plot(train_losses, label="Train Loss")
    ax[1].plot(valid_losses, label="Validation Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    ax[2].plot(valid_epoch_f1, label="Validation F1")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("F1 Score")
    ax[2].set_title("F1 Score over Epochs")
    ax[2].legend()

    plt.savefig(f'./logistic_regression.jpg')

    train_epoch_losses = np.array(train_losses)
    valid_epoch_losses = np.array(valid_losses)
    valid_epoch_accuracy = np.array(valid_accuracies)
    valid_epoch_f1 = np.array(valid_epoch_f1)

    np.savetxt('train_epoch_losses_logistic_regression.txt',
               train_losses, delimiter=',')
    np.savetxt('valid_epoch_losses_logistic_regression.txt',
               valid_losses, delimiter=',')
    np.savetxt('valid_epoch_accuracy_logistic_regression.txt',
               valid_accuracies, delimiter=',')
    np.savetxt('valid_epoch_f1_logistic_regression.txt',
               valid_epoch_f1, delimiter=',')
    np.savetxt('test_loss_accuracy_f1_logistic_regression.txt',
               np.array([test_loss, test_accuracy, test_f1]))
