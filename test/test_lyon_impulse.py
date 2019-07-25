from lyon.calc import LyonCalc
import numpy as np
import matplotlib.pyplot as plt
import pdb

calc = LyonCalc()

waveform = np.asarray([1]+[0]*255, dtype=np.float64)
output = calc.lyon_passive_ear(waveform, sample_rate=16000, decimation_factor=1, ear_q=8, step_factor=0.25)
output = np.clip(output, 0, 0.0004)

plt.imshow(output.T)

plt.savefig('./figure/lyon_impulse.png')

plt.show()








