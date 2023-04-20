import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

import numpy as np
import random
import pandas as pd


from dataset import ImageFileCsvLabelDataset
from discriminator import Discriminator

# Training settings
import argparse
parser = argparse.ArgumentParser(description='testCOVIDXrayData')
parser.add_argument('--explain', type=str, default='testCOVIDXrayDataRemainCSV')
parser.add_argument('--testSet', type=str, default='OR', help='choose testSet')
parser.add_argument('--data', type=str, default='covidXrayData')
parser.add_argument('--dataDir', type=str, default='data/covidXrayData')
parser.add_argument('--weightDir', type=str, default='result_exp/resultWeight_trainingCOVIDXrayData_230404_150944_covidXrayData_OR/mobileNet_epoch_9.pt')
parser.add_argument('--testBatchSize', type=int, default=20, help='test batch size')
parser.add_argument('--classImageSize', type=int, default=224, help='ORN image size')
opt = parser.parse_args()

# pytorch seed setting
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

# csv file load to save
answerCSV = pd.read_csv(opt.dataDir + "/submission.csv")

print(answerCSV)

# data transform
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
# testset_OR = torchvision.datasets.DatasetFolder(root=opt.dataDir + '/test', transform=transform_data_TS, loader=self.root)
# testset_OR = torchvision.datasets.ImageFolder(root=opt.dataDir + '/test', transform=transform_data_TS)

testset_OR = ImageFileCsvLabelDataset(opt.dataDir + '/test/labels.csv', opt.dataDir + '/test', transform=transform_data_TS,)

testloader_OR = torch.utils.data.DataLoader(
    testset_OR,
    batch_size=opt.testBatchSize,
    shuffle=False,
    num_workers=0,
    worker_init_fn=np.random.seed(0))

# discriminator setting
discriminator = Discriminator(numclasses = 1).cuda()
discriminator.load_state_dict(torch.load(opt.weightDir)['discriminator_state_dict'])
discriminator.to(device)
discriminator.eval()

for i, data in enumerate(testloader_OR, 0):
    imgs = data
    aux_d = discriminator(imgs.cuda())

    # _, pred_d = torch.max(aux_d.data, 1)
    pred_d = torch.sigmoid(aux_d.data).round()

    # running_corrects_d += torch.sum(pred_d == labels.data).tolist()
    # running_corrects_d += torch.sum(pred_d == labels.data)
    for j, answer in enumerate(pred_d):
        
        if int(answer) == 1:
            answerCSV.iloc[i*opt.testBatchSize + j, 1] = 'covid'
        elif int(answer) == 0:
            answerCSV.iloc[i*opt.testBatchSize + j, 1] = 'normal'

print(answerCSV)

# csv file save
answerCSV.to_csv(opt.dataDir + "/submission_tested.csv",index=False)

print('Finish Testing!')