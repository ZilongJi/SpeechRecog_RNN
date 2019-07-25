import os
import os.path as osp
import shutil
import sys
import pdb

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import NullFormatter
from matplotlib import offsetbox

import librosa

def tsne_visual(features, labels, epoch, n_components=2):
    """Visualize high-dimensional data in the feature space with t-distributed Stochastic Neighbor Embedding.
    
    Plotting part is from https://github.com/kevinzakka/tsne-viz/blob/master/main.py
    
    Args:
        features: features of all data points. shape (n_samples, n_features)
        labels: labels corresponding to all data points. shape (n_samples)
        epoch: current testing epoch
        n_components: Dimension of the embedded space.
    Returns:
        feat_embedded: Embedding of the training data in low-dimensional space. shape (n_samples, n_components)
    """ 
      
    #feat_embedded = TSNE(n_components=n_components).fit_transform(features)
    
    feat_embedded = PCA(n_components=n_components).fit_transform(features)
    
    classes = np.unique(labels)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #colors = cm.Spectral(np.linspace(0, 1, len(classes)))
    
    xx = feat_embedded[:, 0]
    yy = feat_embedded[:, 1]
    
    #plot the images
    for i, class_i in enumerate(classes.tolist()):
        #ax.scatter(xx[labels==class_i], yy[labels==class_i], color=colors[i], label=str(class_i), s=20)
        ax.scatter(xx[labels==class_i], yy[labels==class_i], label=str(class_i), s=20)
    
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.legend(loc='best', scatterpoints=3, fontsize=20)
    plt.savefig('./pca_'+str(epoch)+'2.png', dpi=500)

root_dir = '/home/jizilong/Desktop/SpeechRecog_Project/Dataset/cut_all'

folders = os.listdir(osp.join(root_dir, 'select2'))

WAV = []
DATA = []
LABELS = []

lll = 0

for fs in folders:
    filenames = os.listdir(osp.join(root_dir, 'select2', fs))
    
    lll += 1
    
    for i, f_ in enumerate(filenames):
        print(i)
        full_f_ = osp.join(root_dir, 'select', fs, f_)
        wav, sample_rate = librosa.load(full_f_, sr=16000)
        S = librosa.feature.melspectrogram(wav, sr=sample_rate, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        mfcc_flatten = mfcc.reshape(-1)
        
        LABELS.append(lll)
        WAV.append(wav)
        DATA.append(mfcc_flatten)

FEAT = np.zeros((len(DATA),208))

for i in range(len(DATA)):
    FEAT[i] = DATA[i]

LABELS = np.asarray(LABELS)

pdb.set_trace()

tsne_visual(FEAT, LABELS, 0, n_components=2)


        
        
        
        
        
        
        
        
        
        
        
        
        
