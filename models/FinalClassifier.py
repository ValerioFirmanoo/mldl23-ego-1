from torch import nn
from models import I3D


class Classifier(nn.Module):
    def __init__(self, num_classes=8): #DOVE VIENE CHIAMATO QUESTO INIT? #num classes qua o glielo passiamo da fuori?
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

        self.classifier = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.classifier(x), {}
