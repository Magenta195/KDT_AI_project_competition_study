import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

import numpy as np
import random

import time
import os

from dataset import ImageFileCsvLabelDataset
from discriminator import Discriminator

# Training settings
import argparse
parser = argparse.ArgumentParser(description='trainingCOVIDXrayData')
parser.add_argument('--explain', type=str, default='trainingCOVIDXrayData')
parser.add_argument('--trainSet', type=str, default='OR', help='choose trainSets for training_ELR')
parser.add_argument('--data', type=str, default='covidXrayData')
parser.add_argument('--dataDir', type=str, default='data/covidXrayData')
parser.add_argument('--trainBatchSize', type=int, default=20, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=20, help='test batch size')
parser.add_argument('--classImageSize', type=int, default=224, help='ORN image size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
opt = parser.parse_args()

# information for save
# thisPath = os.path.dirname(os.path.realpath(__file__))
timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())

path_tar = './result_exp/resultWeight_%s_%s_%s_%s' % \
    (opt.explain ,timestamp, opt.data, opt.trainSet)


os.makedirs(path_tar)
baseName = '%s\mobileNet_' % path_tar 
logName = baseName + opt.data + '_result'
filename_log = '%s.log' % (logName)
f = open(filename_log, 'w')

with open(__file__,'rt',encoding='UTF8') as fi: 
    f.write('\n'.join(fi.read().split('\n')[0:]))

# pytorch seed setting
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

# data transform
transform_data_TR = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.Resize(opt.classImageSize),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(75),
    # transforms.RandomGrayscale(p=0.1),
    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_data_TS = transforms.Compose([
    transforms.Resize(opt.classImageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# cuda setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# dataset and dataloader setting
trainset_OR = ImageFileCsvLabelDataset(opt.dataDir + '/train/labels.csv', opt.dataDir + '/train', 'train', transform=transform_data_TR)

trainset_OR, validationset_OR = torch.utils.data.random_split(trainset_OR, [1800, 200])

trainset = torch.utils.data.ConcatDataset([trainset_OR])

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=opt.trainBatchSize,
    shuffle=True,
    num_workers=0,
    worker_init_fn=np.random.seed(0),
    pin_memory=True
)

valloader = torch.utils.data.DataLoader(
    validationset_OR,
    batch_size=opt.trainBatchSize,
    shuffle=True,
    num_workers=0,
    worker_init_fn=np.random.seed(0),
    pin_memory=True
)

# discriminator setting
discriminator = Discriminator(numclasses = 1).cuda()
discriminator.to(device)
discriminator.eval()

# loss setting
criterion_d = nn.BCEWithLogitsLoss().cuda()

# optimizer setting
optimizer_d = optim.Adam(discriminator.parameters())

# score for saving
best_score = 0

for epoch in range(opt.nEpochs):  # loop over the dataset multiple times
    running_loss_d = 0.0    
    running_corrects_d = 0
    discriminator.train()
    
    for i, data in enumerate(trainloader, 0):
        
        imgs, labels = data

        imgs = imgs.cuda()
        labels = labels.cuda()
        labels = labels.float().unsqueeze(1)
        
        optimizer_d.zero_grad()
        aux_d = discriminator(imgs)
      
        loss_d = criterion_d(aux_d, labels)
        loss_d.backward()
        optimizer_d.step()
        
        # _, pred_d = torch.max(aux_d.data, 1)
        pred_d = torch.sigmoid(aux_d.data).round()
                
        # running_corrects_d += torch.sum(pred_d == labels.data).tolist()
        running_corrects_d += torch.sum(pred_d == labels.data)
        
        running_loss_d = loss_d.item()
        
        # recording
        disp_str = 'Train [%d, %d] d_loss: %.3f' % (epoch, i, running_loss_d)
        print(disp_str)
        f.write(disp_str)
        f.write('\n')
        disp_str = 'Train [%d, %d]  Data(OR) : %d / %d' % (epoch, i, running_corrects_d, (i+1)*opt.trainBatchSize)
        
        print(disp_str)
        f.write(disp_str)
        f.write('\n')
    
    outFileName = baseName + 'epoch_' + str(epoch) +'.pt'
    
    discriminator.eval()
    running_corrects_d = 0

    for it, data in enumerate(valloader, 0):
        imgs, labels = data
        labels = labels.cuda()
        labels = labels.float().unsqueeze(1)
        aux_d = discriminator(imgs.cuda())

        # _, pred_d = torch.max(aux_d.data, 1)
        pred_d = torch.sigmoid(aux_d.data).round()

        # running_corrects_d += torch.sum(pred_d == labels.data).tolist()
        running_corrects_d += torch.sum(pred_d == labels.data)
    
    # recording
    disp_str = 'Val [%d]  OR : %d / %d' % (epoch, running_corrects_d, len(validationset_OR))
    print(disp_str)
    f.write(disp_str)
    f.write('\n')
    
    if running_corrects_d > best_score:
        best_score = running_corrects_d
        best_epoch = epoch
    # save weight    
    print("Save weight : %s" % outFileName)
    
    torch.save({
            'epoch': epoch,
            #'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            #'optimizer_g_state_dict': optimizer_g.state_dict(),            
            'loss_d': loss_d,
            #'loss_g': loss_g,
    }, outFileName)

disp_str = 'Best Epooch is [%d] epoch : %d / %d' % (epoch, running_corrects_d, len(validationset_OR))
print(disp_str)
f.write(disp_str)
f.write('\n')
        
print('Finished Training')