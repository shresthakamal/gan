import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 75

IMG_SHAPE = (1, 28, 28)
IMG_SIZE = 1
for x in IMG_SHAPE:
    IMG_SIZE *= x


class GAN(torch.nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, IMG_SIZE),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(IMG_SIZE, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def generator_forward(self, z):
        return self.generator(z)

    def discriminator_forward(self, img):
        pred = self.discriminator(img)
        return pred.view(-1)
