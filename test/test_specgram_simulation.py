from scipy.io import wavfile 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb

def log_specgram(audio, sample_rate, window_size=10, step_size=5, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps) 

A = np.sin(np.arange(0,5000)/16000*2*np.pi*800)
B = np.sin(np.arange(0,5000)/16000*2*np.pi*2000)
C = np.sin(np.arange(0,5000)/16000*2*np.pi*4000)
waveform = np.asarray(A+B+C, dtype=np.float64)

_, _, specgram = log_specgram(waveform, sample_rate=16000)

plt.figure(figsize=(10,8), dpi=100)

plt.subplot(2,1,1)
plt.plot(np.arange(len(waveform[0:100])), waveform[0:100])
plt.xlabel('Time', fontsize=15)
plt.ylabel('Amp', fontsize=15)

plt.subplot(2,1,2)
plt.imshow(specgram.T, origin='lower', aspect='auto')
plt.xlabel('Time', fontsize=15)
plt.ylabel('HZ', fontsize=15)

plt.savefig('./figure/specgram_multisin.png')

plt.show()
