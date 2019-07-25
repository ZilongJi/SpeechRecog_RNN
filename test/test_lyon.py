from lyon.calc import LyonCalc
import librosa 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb

#filename = '../SpeechRecog_CNN/Dataset/cut/audio/up/571c044e_nohash_0.wav'
filename = '../../SpeechRecog/SpeechRecog_CNN/Dataset/cut/audio/zero/981e2a16_nohash_1.wav'

calc = LyonCalc()

wav, sample_rate = librosa.load(filename, sr=16000)
waveform = np.asarray(wav, dtype=np.float64)

lyon_output = calc.lyon_passive_ear(waveform, sample_rate, decimation_factor=64, ear_q=8, step_factor=0.5)
pdb.set_trace()
print(lyon_output.shape)

#plot
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of zero/981e2a16_nohash_1.wav')
ax1.set_ylabel('Amplitude', fontsize=15)
ax1.plot(np.linspace(0, len(wav)/sample_rate, len(wav)), wav)

ax2 = fig.add_subplot(212)
ax2.imshow(lyon_output.T, aspect='auto')
ax2.set_title('Cochlea Response of zero/981e2a16_nohash_1.wav')
ax2.set_ylabel('Auditory Nerve Place', fontsize=15)
ax2.set_xlabel('Time', fontsize=15)  
           
plt.savefig('./figure/lyon_wav.png')

plt.show()

pdb.set_trace()










