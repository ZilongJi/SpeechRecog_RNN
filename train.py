from __future__ import print_function, absolute_import
import argparse
import time
import pdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torchsummary import summary

from code.utils import *
from code.dataset import KWSData
from code.Model import SimpleLSTM, VallinaRNN, SimpleGRU, LinearClassifier

def get_data(args):

#    mfcc_transform = transforms.Compose([
#        transforms.Normalize(mean=[-0.2107], std=[6.6551])])    
    
    train_set = KWSData(args.data_dir, 
        mode='train', 
        sample_rate=args.sample_rate, 
        new_sample_rate=args.new_sample_rate,
        select_class=None,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
        transform=None, 
        frontend=args.frontend)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    
    validate_set = KWSData(args.data_dir, 
        mode='validate', 
        sample_rate=args.sample_rate, 
        new_sample_rate=args.new_sample_rate,
        select_class=train_set.select_class,
        num_samples=10, 
        transform=None, 
        frontend=args.frontend) 
    val_loader = DataLoader(validate_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    test_set = KWSData(args.data_dir, 
        mode='test', 
        sample_rate=args.sample_rate, 
        new_sample_rate=args.new_sample_rate,
        select_class=train_set.select_class,
        num_samples=100,
        #transform=transforms.Compose([transforms.ToTensor()])) 
        transform=None,
        frontend=args.frontend)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader, test_loader
   
def train(args, model, device, optimizer, loss_func, train_loader, val_loader, test_loader):
    
    #start training
    for epoch in range(args.epochs):
        
        #Adjust learning rate
        adjust_lr_exp(
            optimizer,
            args.base_lr,
            epoch+1,
            args.epochs,
            args.exp_decay_at_epoch) 
        
        model.train()
        
        loss_meter = AverageMeter()
        
        ep_st = time.time()  
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            print(data.shape)
            pdb.set_trace()
            optimizer.zero_grad()
            preds = model(data)
            #pdb.set_trace()
            
            loss = loss_func(preds, target).to(device)
            
            logits = preds.argmax(dim=1, keepdim=True)
            correct += logits.eq(target.view_as(logits)).sum().item()
            
            loss.backward() 
            
            optimizer.step()
            
            loss_meter.update(loss)
       
        #Epoch log
        time_log =  'Ep {}, {:.2f}s'.format(epoch, time.time() - ep_st, )
        
        loss_log = (', loss {:.4f}, acc {}/{} ({:.0f}%)'.format(loss_meter.avg, correct, len(train_loader.dataset), 100.*correct/len(train_loader.dataset)))  
        
        total_log = time_log + loss_log
      
        print(total_log)  
        
        #validation 
        val_acc = test(args, model, device, loss_func, val_loader)
            
    #adjust learning rate back to initialized learning rate
    print('Learning rate adjuested back to base learning rate {:.10f}'.format(args.base_lr))
    for g in optimizer.param_groups:
        g['lr'] = args.base_lr             
        
    #test
    test_acc = test(args, model, device, loss_func, test_loader) 
    return test_acc          

def test(args, model, device, loss_func, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            preds = model(data)
            test_loss += loss_func(preds, target).item()
            preds = preds.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
100. * correct / len(test_loader.dataset))) 

    return correct*1. / len(test_loader.dataset)

def main(args, HU):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu") 
     
    train_loader, val_loader, test_loader = get_data(args)  
    
    loss_func = nn.CrossEntropyLoss()
    
    #model = SimpleGRU(args, num_classes=args.num_classes, hidden_units=HU, hidden_layers=1).to(device)
    
    model = SimpleLSTM(args, num_classes=args.num_classes, hidden_units=HU, hidden_layers=1).to(device)
    
    #model = LinearClassifier(num_classes=args.num_classes).to(device)
    
    #summary(model, (16,128))
    
    #optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    
    test_acc = train(args, model, device, optimizer, loss_func, train_loader, val_loader, test_loader)

    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Key Words Speech Recognition Task')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='../SpeechRecog/SpeechRecog_CNN/Dataset/cut')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train in a episode (default: 10)')                        
    parser.add_argument('--exp_decay_at_epoch', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')  
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')                        
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') 
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='original sampling rate')
    parser.add_argument('--new_sample_rate', type=int, default=8000,
                        help='new sampling rate')                
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')                         
    parser.add_argument('--frontend', type=str, default='mfcc_delta')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10)
                                                                    
    args = parser.parse_args()  
    
    episodes=20
    #HU = [5,10,15,20,25,30,35,40,45,50]
    #HU = [50, 200, 500, 1000]
    HU = [50]
    ALL_ACC = []
    MEAN_ACC = []
    STD_ACC = []
    for hu in HU:
        ACC = []
        for episode in range(episodes):
            print("~"*50)
            print("~"*50)
            print('Current training episode is {}, hidden units number is {}'.format(episode,hu))
            print("~"*50)
            print("~"*50)
            acc = main(args, hu)                        
            ACC.append(acc)
            
        mean_acc = np.mean(ACC)
        std_acc = np.std(ACC)
        MEAN_ACC.append(mean_acc)
        STD_ACC.append(std_acc)
        print(ACC)
        print('mean acc is ', mean_acc)
        print('std acc is ', std_acc)
            
        ALL_ACC.append(ACC)
    
    print(ALL_ACC)   
    print(MEAN_ACC)
    print(STD_ACC)     
        

              
                        
                        
                        
                        
                        
                        
                              
