from scipy.io import wavfile 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb
import librosa
from librosa import display

#filename = '../SpeechRecog_CNN/Dataset/cut/audio/up/571c044e_nohash_0.wav'
filename = '../../SpeechRecog/SpeechRecog_CNN/Dataset/cut/audio/zero/981e2a16_nohash_1.wav'

#sample_rate1, wav1 = wavfile.read(filename)
#wav1 = np.float32(wav1)
#S1 = librosa.feature.melspectrogram(wav1, sr=sample_rate1, n_mels=128)
#log_S1 = librosa.power_to_db(S1, ref=np.max)
#mfcc1 = librosa.feature.mfcc(S=log_S1, n_mfcc=13)

wav, sample_rate = librosa.load(filename, sr=16000)
S = librosa.feature.melspectrogram(wav, sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
#mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

#plot
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of zero/981e2a16_nohash_1.wav')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, len(wav)/sample_rate, len(wav)), wav)

ax2 = fig.add_subplot(212)
display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
plt.title("Mel power spectrogram")
plt.colorbar(format='%+02.0f dB')
           
plt.savefig('./figure/melspecgram_wav.png')
plt.show()
