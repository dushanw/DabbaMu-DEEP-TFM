
from torch import nn
class simple_generator(nn.Module):
    def __init__(self, T, img_size=32):
        super(simple_generator, self).__init__()
        self.T= T
        self.img_size= img_size
        self.model= nn.Sequential(
            nn.Conv2d(self.T, 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(3, 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x= x.view(-1, self.T, self.img_size, self.img_size)
        return self.model(x).view(-1, 1, self.img_size, self.img_size)