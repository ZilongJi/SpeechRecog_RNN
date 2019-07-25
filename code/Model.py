import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
import pdb

class Resnet18(nn.Module):
    def __init__(self, num_classes=None):
        super(Resnet18, self).__init__() 
        self.base = resnet18(pretrained=False)
        self.fc1 = nn.Linear(512, 128)
        self.global_bn = nn.BatchNorm1d(128) 
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.base(x)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.global_bn(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
        
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=None):
        super(SimpleCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout()) 
        
        def fc_block(in_channels, out_channels):
            return nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.Linear(in_channels, out_channels),
                nn.ReLU())
                
        self.encoder = nn.Sequential(
            conv_block(1, 8),
            conv_block(8, 16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(),
            Flatten(),
            fc_block(3840, 256),
            fc_block(256, 128),
            nn.Linear(128, num_classes)    
        )
    def forward(self, x):
        x = self.encoder(x)
        return x
        
class SimpleLSTM(nn.Module):
    def __init__(self, args, num_classes=None, hidden_units=None, hidden_layers=None):
        super(SimpleLSTM, self).__init__()
        #Define the LSTM layer, batch_first=True means input and output tensors are provided as (batch, seq, feature)
        if args.frontend=='specgram':
            IS=161
        elif args.frontend=='melspecgram':
            IS=128
        elif args.frontend=='mfcc':
            IS=13
        elif args.frontend=='mfcc_delta':
            IS=13 
        elif args.frontend=='mfcc_all':
            IS=26  
        elif args.frontend=='lyon':
            IS=41
        else:
            raise RuntimeError("Use the correct front end!")   
        self.lstm = nn.LSTM(input_size=IS, hidden_size=hidden_units, num_layers=hidden_layers, batch_first=True) 
        #Define the output layer
        self.linear = nn.Linear(hidden_units, num_classes)
        
    def forward(self, x):
        #pdb.set_trace()
        #dim = [batch, seq, hidden_size]
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:,-1,:]
        output = self.linear(last_out) #dim=[batch, num_classes]
        return output

class VallinaRNN(nn.Module):
    def __init__(self, args, num_classes=None, hidden_units=None):
        super(VallinaRNN, self).__init__()
        #Define the Vallina RNN layer, batch_first=True means input and output tensors are provided as (batch, seq, feature)
        if args.frontend=='specgram':
            IS=81
        elif args.frontend=='melspecgram':
            IS=128
        elif args.frontend=='mfcc':
            IS=13
        elif args.frontend=='mfcc_delta':
            IS=13       
        elif args.frontend=='mfcc_all':
            IS=26       
        elif args.frontend=='lyon':
            IS=41
        else:
            raise RuntimeError("Use the correct front end!")   
        self.rnn = nn.RNN(input_size=IS, hidden_size=hidden_units, num_layers=1, batch_first=True)
        #Define the output layer
        self.linear = nn.Linear(hidden_units, num_classes)        

    def forward(self, x):
        #pdb.set_trace()
        #dim = [batch, seq, hidden_size]
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:,-1,:]
        output = self.linear(last_out) #dim=[batch, num_classes]
        return output

class SimpleGRU(nn.Module):
    def __init__(self, args, num_classes=None, hidden_units=None, hidden_layers=None):
        super(SimpleGRU, self).__init__()
        #Define the Vallina RNN layer, batch_first=True means input and output tensors are provided as (batch, seq, feature)
        if args.frontend=='specgram':
            IS=81
        elif args.frontend=='melspecgram':
            IS=128
        elif args.frontend=='mfcc':
            IS=13
        elif args.frontend=='mfcc_delta':
            IS=13 
        elif args.frontend=='mfcc_all':
            IS=26             
        elif args.frontend=='lyon':
            IS=41
        else:
            raise RuntimeError("Use the correct front end!")   
        self.gru = nn.GRU(input_size=IS, hidden_size=hidden_units, num_layers=hidden_layers, batch_first=True)
        #Define the output layer
        self.linear = nn.Linear(hidden_units, num_classes)        

    def forward(self, x):
        #pdb.set_trace()
        #dim = [batch, seq, hidden_size]
        gru_out, _ = self.gru(x)
        last_out = gru_out[:,-1,:]
        output = self.linear(last_out) #dim=[batch, num_classes]
        return output  
        
class LinearClassifier(nn.Module):
    def __init__(self, num_classes=None):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(13*16, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x



















