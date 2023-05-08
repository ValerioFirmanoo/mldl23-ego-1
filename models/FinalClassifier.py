from torch import nn
from models import I3D


class Classifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.avg_modality = model_args.avg_modality
        self.num_classes = model_args.num_classes
        self.num_clips = model_args.num_clips

        self.AvgPool = nn.AdaptiveAvgPool2d((1,1024))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),
            nn.ReLU())

        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.AvgPool(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x= self.fc2(x)
        return x, {}
