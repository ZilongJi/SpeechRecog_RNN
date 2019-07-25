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


plt.figure(figsize=(10,8), dpi=100)
plt.subplot(2,1,1)
plt.plot(np.arange(len(wav[0:100])), wav[0:100])
plt.xlabel('Time', fontsize=15)
plt.ylabel('Amp', fontsize=15)

plt.subplot(2,1,2)
display.specshow(log_S, sr=16000, x_axis='time', y_axis='mel')
plt.title("Mel power spectrogram")
plt.colorbar(format='%+02.0f dB')
plt.savefig('./figure/melspecgram_multisin.png')
plt.tight_layout()

plt.show()
