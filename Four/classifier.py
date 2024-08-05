import torch.nn as nn
class LicencePlateClassifier(nn.Module):
    def __init__(self, num_classes = 36):
        super().__init__(LicencePlateClassifier,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),


        )

        self.classifier = self.Sequential(
            nn.Flatten(),
            nn.Linear(256*16*16,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256,8)


        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1,8,42)
    
l = LicencePlateClassifier
