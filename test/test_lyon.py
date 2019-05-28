from lyon.calc import LyonCalc
from scipy.io import wavfile 
import numpy as np
from code.utils import pad_audio, chop_audio, log_specgram
from scipy import signal
import pdb

#filename = '../SpeechRecog_CNN/Dataset/cut/audio/up/571c044e_nohash_0.wav'
filename = '../SpeechRecog_CNN/Dataset/cut/audio/zero/981e2a16_nohash_1.wav'
decimation_factor = 64
calc = LyonCalc()

sample_rate, old_wav = wavfile.read(filename)

wav = pad_audio(old_wav, L=8000)
if len(wav)>8000:
    wav = chop_audio(wav, L=8000)
resampled = signal.resample(wav, int(sample_rate / sample_rate * wav.shape[0]))

pdb.set_trace()

waveform1 = np.asarray(old_wav, dtype=np.float64)
output = calc.lyon_passive_ear(waveform1, sample_rate, decimation_factor)

waveform2 = np.asarray(resampled, dtype=np.float64)
output = calc.lyon_passive_ear(waveform2, sample_rate, decimation_factor)










