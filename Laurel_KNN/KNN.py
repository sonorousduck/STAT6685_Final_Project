
import os

import numpy as np
import pandas as pd

import librosa
import librosa.display
import soundfile as sf # librosa fails when reading files on Kaggle.
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_recall_fscore_support



train_csv = pd.read_csv('D:/DQN_Stuff/KNN/train_metadata.csv')
# Plot the sample.

print (train_csv.head())
# We can even display an spectogram of the mfccs.
#librosa.display.specshow(mfccs, sr=sr, x_axis='time')
#plt.show()
#labels = ["akiapo","aniani","apapan","barpet","crehon","elepai","ercfra","hawama","hawcre","hawgoo","hawhaw","hawpet1","houfin","iiwi","jabwar","maupar","omao","puaioh","skylar","warwhe1","yefcan",]
#train_csv = train_csv[train_csv.primary_label.isin(labels)]
print(train_csv.shape)
train_csv = train_csv.drop(3502)
#v = train_csv.primary_label.value_counts()
#train_csv = train_csv[train_csv.primary_label.isin(v.index[v.gt(50)])]
print(train_csv.shape)
def mean_mfccs(x, index, train_csv):
    try:
    
        return [np.mean(feature) for feature in librosa.feature.mfcc(y=x)]
    except:
          train_csv = train_csv.drop(labels=index, axis=0)
          print("An exception occurred at index ", index)
          return "error" 

          

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]] 

def get_audios():
    train_path = "D:/DQN_Stuff/KNN/train/"
   
    samples = []
    for index, row in train_csv.iterrows():
        if index %100 == 0:
            print(index)
        #print(row['filename'])
        file_name = row['filename']
        x, sr = sf.read(train_path + file_name, always_2d=True)
        x = parse_audio(x)
        m = mean_mfccs(x, index, train_csv)
        if m != "error":
            samples.append(m)
    #print(np.array(samples))   
    return np.array(samples) 

def get_samples():
    audio = get_audios()
    values = train_csv['primary_label'].values
    print(len(values))
    print(len(audio))
    return audio, values


X, Y = get_samples()

x_train, x_test, y_train, y_test = train_test_split(X, Y)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA().fit(x_train_scaled)


grid_params = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled, y_train)

print(f'Model Score: {model.score(x_test_scaled, y_test)}')

y_predict = model.predict(x_test_scaled)
print(model.best_params_)
print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')
print("Model f1 score = ", f1_score(y_test, y_predict, average=None))
with open('Knn_model_lame.pkl','wb') as f:
    pickle.dump(model,f)

print (precision_recall_fscore_support(y_test, y_predict, average="weighted"))

