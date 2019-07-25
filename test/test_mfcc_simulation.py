from scipy.io import wavfile 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb
import librosa
from librosa import display

A = np.sin(np.arange(0,5000)/16000*2*np.pi*800)
B = np.sin(np.arange(0,5000)/16000*2*np.pi*2000)
C = np.sin(np.arange(0,5000)/16000*2*np.pi*4000)
wav = np.asarray(A+B+C, dtype=np.float64)

S = librosa.feature.melspectrogram(wav, sr=16000, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

plt.figure(figsize=(10,8), dpi=100)
plt.subplot(2,1,1)
plt.plot(np.arange(len(wav[0:100])), wav[0:100])
plt.xlabel('Time', fontsize=15)
plt.ylabel('Amp', fontsize=15)

plt.subplot(2,1,2)
display.specshow(mfcc, sr=16000, x_axis='time', y_axis='mel')
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')     
plt.title('MFCC')
plt.colorbar()    
plt.savefig('./figure/mfcc_simulation.png')
plt.show()
