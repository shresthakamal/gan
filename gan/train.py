import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from gan.architecture.nn import nnGAN

random_seed = 123
generator_learning_rate = 0.001
discriminator_learning_rate = 0.001
LATENT_DIM = 75
BATCH_SIZE = 128
NUM_EPOCHS = 10
IMG_SHAPE = (1, 28, 28)
IMG_SIZE = 1
for x in IMG_SHAPE:
    IMG_SIZE *= x


def train(start, device):
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize tensorboard writer
    writer = SummaryWriter(f"runs/GAN_MNIST/{time.time()}")

    # get the mnist data from datasets.MNIST
    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transforms.ToTensor(), download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(random_seed)

    model = nnGAN()
    model = model.to(device)

    optim_G = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
    optim_D = torch.optim.Adam(
        model.discriminator.parameters(), lr=discriminator_learning_rate
    )

    for epoch in range(NUM_EPOCHS):
        model = model.train()

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = (features - 0.5) * 2.0
            features = features.view(-1, IMG_SIZE).to(device)
            targets = targets.to(device)

            valid = torch.ones(features.size(0)).float().to(device)
            fake = torch.zeros(features.size(0)).float().to(device)

            # Training Generator
            # creating noise
            z = torch.zeros(targets.size(0), LATENT_DIM).uniform_(-1, 1).to(device)

            # generating images
            generated_featuers = model.generator_forward(z)
            discriminator_pred = model.discriminator_forward(generated_featuers)

            # calculating loss
            g_loss = torch.nn.BCELoss()(discriminator_pred, valid)

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            # Train the discriminator
            discriminator_pred_real = model.discriminator_forward(
                features.view(-1, IMG_SIZE)
            )
            real_loss = torch.nn.BCELoss()(discriminator_pred_real, valid)

            discriminator_pred_fake = model.discriminator_forward(
                generated_featuers.detach()
            )
            fake_loss = torch.nn.BCELoss()(discriminator_pred_fake, fake)

            d_loss = (real_loss + fake_loss) / 2

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # plot losses based on batch and epoch
            writer.add_scalars(
                "Loss",
                {"g_loss": g_loss, "d_loss": d_loss},
                epoch * len(train_loader) + batch_idx,
            )

            ### LOGGING
            if not batch_idx % 100:
                print(
                    "Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f"
                    % (
                        epoch + 1,
                        NUM_EPOCHS,
                        batch_idx,
                        len(train_loader),
                        g_loss,
                        d_loss,
                    )
                )

        print("Time elapsed: %.2f min" % ((time.time() - start) / 60))

    print("Total Training Time: %.2f min" % ((time.time() - start) / 60))

    # save the model
    torch.save(model.state_dict(), "gan_nn_mnist.pth")


# python main block
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    train(start, device)
