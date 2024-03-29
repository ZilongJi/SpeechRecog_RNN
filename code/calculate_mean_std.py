from __future__ import division
import os
import pdb
import os.path as osp
import numpy as np
from scipy.io import wavfile 
from scipy import signal
import librosa
from utils import pad_audio, chop_audio, log_specgram

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def select_samples(data_base, select_class, num_samples):
    #create a dict to reform data_base
    new_data_base = {}    
    for class_k in select_class:
        new_data_base[class_k] = []    
    for item_ in data_base:
        class_g = item_.split('/')[0]
        if class_g in select_class:
            new_data_base[class_g].append(item_)
    
    data_list = []      
    for class_i in select_class:
        class_i_base = new_data_base[class_i]
        N = len(class_i_base)
        if num_samples==None:
            idx = np.random.choice(N, N, replace=False)
        else:
            idx = np.random.choice(N, num_samples, replace=False)
        class_i_samples = [class_i_base[i] for i in idx]
        data_list += class_i_samples
        
    return data_list

class KWSData(Dataset):
    def __init__(self, root_dir, mode, sample_rate, new_sample_rate, transform=None, select_class=None, num_classes=10, num_samples=10, mfcc=False):
        super(KWSData, self).__init__()
        assert mode == 'train' or mode == 'validate' or mode == 'test'
        
        with open(osp.join(root_dir, 'train_file.txt'), 'r') as f:
            self.train_base = f.read().splitlines()
        with open(osp.join(root_dir, 'validate_file.txt'), 'r') as f:
            self.val_base = f.read().splitlines()
        with open(osp.join(root_dir, 'test_file.txt'), 'r') as f:
            self.test_base = f.read().splitlines()
        
        self.mode = mode
        
        class_list = os.listdir(osp.join(root_dir, 'audio'))
        class_list.sort()
        remove_list = []
        
        for i in range(len(class_list)):
            if class_list[i][0] == '.' or class_list[i][0] == '_':
                remove_list.append(class_list[i])
                
        for rl in remove_list:
            class_list.remove(rl)  
            
        #random sampling num_classes classes for training
        if mode=='train':
            N = len(class_list)
            index = np.random.choice(N, num_classes, replace=False)
            select_class = [class_list[i] for i in index]
        else:
            select_class=select_class

        #pdb.set_trace()
        #random sampling num_samples samples from selected classes
        if mode=='train':
            self.data_list = select_samples(self.train_base, select_class, num_samples)
        elif mode=='validate':
            self.data_list = select_samples(self.val_base, select_class, num_samples)
        else:
            self.data_list = select_samples(self.test_base, select_class, num_samples)

        self.class2num = {c: n for n, c in enumerate(select_class)} 
        self.n_class = len(select_class)
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.new_sample_rate = new_sample_rate
        self.select_class=select_class
        self.mfcc = mfcc
        self.transform = transform

    def __getitem__(self, index):
        sample_rate, wav = wavfile.read(osp.join(self.root_dir, 'audio', self.data_list[index]))
        
        #if length < 16000, then pad the original wav
        wav = pad_audio(wav)

        if len(wav)>16000:
            wav = chop_audio(wav)
        resampled = signal.resample(wav, int(self.new_sample_rate / self.sample_rate * wav.shape[0]))
        
        #print(resampled.shape)
        
        if not self.mfcc:
            #calculate log specgram and return data
            _, _, specgram = log_specgram(resampled, sample_rate=self.new_sample_rate)
            
            #print(specgram.shape)
            
            label = self.class2num[self.data_list[index].split("/")[0]]        
        
            if self.transform is not None:
                specgram = self.transform(specgram)
            specgram.shape
            #specgram = specgram[np.newaxis,:,:]

            return specgram, label
        else:
            #calculate Mel power spectrogram and MFCC and return data
            #You can calculate Mel power spectrogram and MFCC using for example librosa python package.
            S = librosa.feature.melspectrogram(resampled, sr=self.sample_rate, n_mels=128)
            # Convert to log scale (dB). We'll use the peak power (max) as reference.
            log_S = librosa.power_to_db(S, ref=np.max)
            
            #print(log_S.shape)
            
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            
            # Let's pad on the first and second deltas while we're at it
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            output = delta2_mfcc.T.astype(np.float32)
            
            label = self.class2num[self.data_list[index].split("/")[0]]  
            
            return output, label

    def __len__(self):
        return len(self.data_list)     
           
     
        
if __name__ == '__main__':
    root_dir = '/home/jizilong/Desktop/SpeechRecog/SpeechRecog_CNN/Dataset'
    sample_rate = 16000
    new_sample_rate = 8000
    transform = None
    batch_size = 20
    
#    pdb.set_trace()
    
    data_set = KWSData(root_dir, 'train', sample_rate, new_sample_rate, transform=None, select_class=None, num_classes=30, num_samples=None, mfcc=True)
    
    dat_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    data_mean, data_std = 0, 0
    
    for i, (data, target) in enumerate(dat_loader):
        print(i)
        data_mean += torch.mean(data)
        data_std += torch.std(data)
    pdb.set_trace()
    
    final_mean = data_mean/(i+1)
    final_std = data_std/(i+1)
        
    
        
        
    
    '''    
    dat_loader = iter(dat_loader)
    A = dat_loader.next()
    '''         






