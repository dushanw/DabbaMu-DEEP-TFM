from torch import nn
import torch

def classification_accuracy(labels, outputs):
    return (torch.argmax(outputs, axis=1) == labels).float().mean().detach().cpu().numpy()

class simple_mnist_classifier(nn.Module):
    def __init__(self, img_size):
        super(simple_mnist_classifier, self).__init__()
        self.img_size= img_size
        self.model = nn.Sequential(
            nn.Linear(img_size**2, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, 10))
    def forward(self, x):
        x= x.view(-1, self.img_size**2)
        return self.model(x)
        