import numpy as np
from scipy import signal
import webrtcvad
import struct
import matplotlib.pyplot as plt
import pdb

def speechdetect(samples, sample_rate):
    #Create a Vad object
    vad = webrtcvad.Vad()
    #set its aggressiveness mode
    vad.set_mode(3)
    
    #convert samples to raw 16 bit per sample stream needed by webrtcvad
    raw_samples = struct.pack("%dh" % len(samples), *samples)
    
    #run the detector on windows of 30 m
    
    window_duration = 0.03 # duration in seconds
    samples_per_window = int(window_duration * sample_rate + 0.5)
    bytes_per_sample = 2
    
    segments = []
    
    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
    
        is_speech = vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], sample_rate = sample_rate)

        segments.append(dict(
            start = start,
            stop = stop,
            is_speech = is_speech))
    
    return segments
    
    '''        
    #plot the range of samples identified as speech in orange   
    plt.figure(figsize = (10,7))
    plt.plot(samples)
    ymax = max(samples)
    # plot segment identifed as speech
    for segment in segments:
        if segment['is_speech']:
            plt.plot([ segment['start'], segment['stop'] - 1], [ymax * 1.1, ymax * 1.1], color = 'orange')

    plt.xlabel('sample')
    plt.grid()  
    
    pdb.set_trace()   
    '''
    
def extract_start(segments):
    start_idx = []
    flag = False #no voice detected, all segments are false
    for segment in segments:
        if segment['is_speech']:
            flag = True
            start_idx.append(segment['start'])
    if not flag:
        start_idx.append(0)
    return start_idx, flag

'''    
def find_longest_duration(segments):
    #find the longest speech duration in a wav
    lll = len(segments)
    
    for i, segment in enumerate(segments):
        start = segment['start']
        stop = segments['stop']
''' 

def find_final_start(segments, samples, win_size):
    start_idx, flag = np.asarray(extract_start(segments))
    #print(start_idx, samples)
    res = running_mean(samples, win_size)
    res_at_idx = res[start_idx]
    max_idx = np.argmax(res_at_idx)
    final_start_idx = start_idx[max_idx] 
    
    return final_start_idx, flag  
           
def running_mean(x, win_size):
    shorter_winsize = int(win_size/2)
    x = pad_audio(x, L=16000+win_size)
    x = np.abs(x)
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return cumsum[shorter_winsize:] - cumsum[:-shorter_winsize]            

def cut_audio(samples, final_start, win_size):
    if final_start+win_size<16000:
        new_samples = samples[final_start:final_start+win_size]
    else:
        new_samples = samples[final_start:]
        new_samples = pad_audio(new_samples, L=win_size)
    return new_samples
 
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

def pad_audio(samples, L=16000):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(0, L - len(samples)), mode='constant', constant_values=(0, 0))
    
def chop_audio(samples, L=16000):
    #random select a length L from samples
    beg = np.random.randint(0, len(samples) - L)
    return samples[beg: beg + L]

def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """
    Decay exponentially in the later phase of training. All parameters in the 
    optimizer share the same learning rate.
    """   
    assert ep >= 1, "Current epoch number should be >= 1"  
     
    if ep < start_decay_at_ep:
        #print('=====> lr stays the same as base_lr {:.10f}'.format(base_lr))
        return
    
    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                       / (total_ep + 1 - start_decay_at_ep))))  
    #print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))   

class AverageMeter(object):
    """Modified from Tong Xiao's open-reid. 
    Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)






   
