from torchvision.models import mobilenet_v2
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, numclasses):
        super(Discriminator, self).__init__()
        self.model_ft = mobilenet_v2(pretrained=True)
        
        
        self.model_ft.classifier[1] = nn.Linear(in_features=self.model_ft.classifier[1].in_features, out_features=numclasses)
        # model_ft.classifier[1] = nn.Linear(model_ft.last_channel, numclasses)
        
        # num_ftrs = self.model_ft.last_channel
        # self.model_ft.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(num_ftrs, numclasses, bias=True),
        # )
       
        # self.model_ft.fc = nn.Linear(num_ftrs, numclasses)
         
        # print(self.model_ft)
        
    def forward(self, img):
        # x = self.model_ft.features(img)
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # x = self.model_ft.classifier(x)
        x = self.model_ft(img)
        
        return x