from lyon.calc import LyonCalc
import numpy as np
import matplotlib.pyplot as plt
import pdb

calc = LyonCalc()

#waveform = np.asarray([1]+[0]*255, dtype=np.float64)
#output = calc.lyon_passive_ear(waveform, sample_rate=16000, decimation_factor=1, ear_q=8, step_factor=0.25)
#output = np.clip(output, 0, 0.0004)

#plt.imshow(output.T)
#plt.show()

#waveform = np.asarray(np.sin(np.arange(0,2041)/20000*2*np.pi*800), dtype=np.float64)
#output = calc.lyon_passive_ear(waveform, sample_rate=20000, decimation_factor=20, ear_q=8, step_factor=0.25)

#plt.imshow(output.T)
#plt.show()
A = np.sin(np.arange(0,5000)/16000*2*np.pi*800)
B = np.sin(np.arange(0,5000)/16000*2*np.pi*2000)
C = np.sin(np.arange(0,5000)/16000*2*np.pi*4000)
waveform = np.asarray(A+B+C, dtype=np.float64)
output = calc.lyon_passive_ear(waveform, sample_rate=16000, decimation_factor=20, ear_q=8, step_factor=0.25)

plt.figure(figsize=(10,8), dpi=100)

plt.subplot(2,1,1)
plt.plot(np.arange(len(waveform[0:100])), waveform[0:100])
plt.xlabel('Time', fontsize=15)
plt.ylabel('Amp', fontsize=15)

plt.subplot(2,1,2)
plt.imshow(output.T, aspect='auto')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Auditory Nerve Place', fontsize=15)

plt.savefig('./figure/lyon_multisin.png')

plt.show()








