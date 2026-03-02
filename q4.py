import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torchvision
import logging

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import time
from IPython.display import display, clear_output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
NoneType = type(None)

# You can copy this code to your personal pipeline project or execute it here.
class Generator(nn.Module):
    """
    Generator class for the GAN
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


# You can copy this code to your personal pipeline project or execute it here.
class Discriminator(nn.Module):
    """
    Discriminator class for the GAN
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output


# You can copy this code to your personal pipeline project or execute it here.
def train_gan(
    batch_size: int = 32,
    num_epochs: int = 100,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    """
    The method trains a Generative Adversarial Network and is based on:
    https://realpython.com/generative-adversarial-networks/

    The Generator network tries to generate convincing images of handwritten digits.
    The Discriminator needs to detect if the image was created by the Generater or if the image is a real image from
    a known dataset (MNIST).
    If both the Generator and the Discriminator are optimized, the Generator is able to create images that are difficult
    to distinguish from real images. This is goal of a GAN.

    This code produces the expected results at first attempt at about 50 epochs.

    :param batch_size: The number of images to train in one epoch.
    :param num_epochs: The number of epochs to train the gan.
    :param device: The computing device to use. If CUDA is installed and working then `cuda:0` is chosen
        otherwise 'cpu' is chosen. Note: Training a GAN on the CPU is very slow.

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    It contains at least two bugs: one structural bug and one cosmetic bug. Both bugs are from the original tutorial.

    | ``1   Changing the batch_size from 32 to 64 triggers the structural bug.``
    | ``2   Can you also spot the cosmetic bug?``
    | ``Note: to fix this bug a thorough understanding of GANs is not necessary.``

    Change the batch size to 64 to trigger the bug with message:
    ValueError: "Using a target size (torch.Size([128, 1])) that is different to the input size (torch.Size([96, 1])) is deprecated. Please ensure they have the same size."

    >>> train_gan(batch_size=32, num_epochs=100)

    Answer:
    | ``1
        The error occurs when n == 937 at the line `loss_discriminator = loss_function(output_discriminator, all_samples_labels)`,
        suggesting that `loss_function = nn.BCELoss()`
        or `nn.BCELoss()` is triggering this error. Further debugging into variables, I find that the len of all_samples in this scenario is 96.
        In the previous iteration, it was 128. This causes the output_discriminator to have length 96, thus raising the value error.
        The length of real_samples of iteration 937 is 32, instead of 64. Thus the concat `torch.cat()` call combines the two to create.
            - This can be inspected by placing a conditional breakpoint at `for n, (real_samples, mnist_labels) in enumerate(train_loader):`
              The condition being n==935 or n==936.
              At n==935 we can see the size of real_samples to be 64 upon stepping into the loop.
              At n==936 we can see the size of real_samples to be 32 upon stepping into the loop.
            - More into this, the `train_loader/train_set` has 60000 data points.
              When stack_size is set to 64, it cannot evenly distribute it.
              60000/64 = 937.5, which is where the error happens, rounded down.
            - I tried the following to try and make the program run without errors being raised.
              I tried to slice the train_set with the list slicing but it did not work.
              I used torch's subset to slice the training set. No errors are raised, however the images don't render correctly towards the end.
              My other approach would be to raise an error that batch size 64 is unsupported.
              My final approach would be to, instead of slicing/trimming the initial dataset, either double up the final 32 real_samples or drop that iteration entirely.
            - Alternatively, if it is possible to pre-set the number of data points, we could instead reinstatiate the train_set
              by adding the remainder to the previously found number of datapoints. Not sure if possible.



    """
    # Add/adjust code.

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    try:
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform
        )
    except:
        print("Failed to download MNIST, retrying with different URL")
        # see: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        torchvision.datasets.MNIST.resources = [
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
                "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                "d53e105ee54ea40749a09fcbcd1e9432",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                "9fb629c4189551a2d022fa330f9573f3",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
                "ec29112dd5afa0611ce80d1b7f02629c",
            ),
        ]
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform
        )

    # Failed solution. (trimming data set to fix datapoints modulo 64)
        # train_set_points = len(train_set.data)
        # if train_set_points % batch_size != 0:
        #     trim_amount =  train_set_points % batch_size
        #     indices = torch.arange(0, train_set_points - trim_amount)
        #     train_set = data_utils.Subset(train_set, indices) # ?
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # example data
    real_samples, mnist_labels = next(iter(train_loader))

    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis("off")

    fig.tight_layout()
    fig.suptitle("Real images")
    # display(fig)
    # Save the figure instead of displaying
    fig.savefig("images/real_image")
    time.sleep(5)

    # Set up training
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    # train
    for epoch in range(num_epochs):
        logger.info(f"Currently on epoch #{epoch}")
        for n, (real_samples, mnist_labels) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            if len(real_samples) != batch_size:
                batch_size = len(real_samples)
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss and samples generated
            if n == 0 or n == len(train_loader) - 1:
                name = f"Generate images\n Epoch: {epoch} Loss D.: {loss_discriminator:.2f} Loss G.: {loss_generator:.2f}"
                generated_samples = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis("off")
                fig.suptitle(name)
                fig.tight_layout()
                
                # clear_output(wait=False)
                # display(fig)
                # Save the figure instead
                fig.savefig(f"images/fig_{epoch}_{n}")
    return


train_gan(batch_size=64, num_epochs=100)
