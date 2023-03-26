import time

import matplotlib.pyplot as plt
import torch

from gan.architecture.nn import nnGAN

LATENT_DIM = 75


def eval(start, device):
    model = nnGAN()
    # load the model
    model.load_state_dict(torch.load("gan_nn_mnist.pth"))
    model = model.to(device)

    model.eval()

    # generate images
    z = torch.zeros(10, LATENT_DIM).uniform_(-1, 1).to(device)
    generated_featuers = model.generator_forward(z)

    imgs = generated_featuers.view(-1, 1, 28, 28).detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 2.5))

    for i, ax in enumerate(axes):
        # save the image
        ax.imshow(imgs[i].reshape(28, 28), cmap="gray")
    plt.show()

    # save the image based on time
    plt.savefig(f"gan/images/gan_mnist_{start}.png")


# python main block
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    eval(start, device)
