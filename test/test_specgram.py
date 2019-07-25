from scipy.io import wavfile 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb

def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps) 

#filename = '../SpeechRecog_CNN/Dataset/cut/audio/up/571c044e_nohash_0.wav'
filename = '../../SpeechRecog/SpeechRecog_CNN/Dataset/cut/audio/zero/981e2a16_nohash_1.wav'

sample_rate, wav = wavfile.read(filename)
freqs, times, specgram = log_specgram(wav, sample_rate=sample_rate)

#plot
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of zero/981e2a16_nohash_1.wav')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, len(wav)/sample_rate, len(wav)), wav)

ax2 = fig.add_subplot(212)
ax2.imshow(specgram.T, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of zero/981e2a16_nohash_1.wav')
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')  
           
plt.savefig('./figure/specgram_wav.png')
