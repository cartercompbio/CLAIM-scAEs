import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
from functools import reduce

class Encoder(torch.nn.Module):
    def __init__(self, input_shape, hid_size=512, latent_size=10):
        super().__init__()
        #flat_features = reduce(mul, input_shape, 1)
        self._hid = torch.nn.Sequential(
            torch.nn.Linear(input_shape, hid_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_size, hid_size),
            torch.nn.ReLU(),
        )
        self._mean = torch.nn.Linear(hid_size, latent_size)
        self._log_std = torch.nn.Linear(hid_size, latent_size)

    def __call__(self, x):
        # (batch_size, w, d) -> (batch_size, w*d)
        #x = x.view(x.shape[0], -1)
        h0 = self._hid(x)

        mean = self._mean(h0)
        sigma = torch.exp(0.5 * self._log_std(h0))
        return mean, sigma


class Decoder(torch.nn.Module):
    def __init__(self, output_shape, hid_size=512, latent_size=10):
        super().__init__()
        self._output_shape = output_shape
        #flat_features = reduce(mul, output_shape, 1)
        self._layer = torch.nn.Sequential(
            torch.nn.Linear(latent_size, hid_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_size, output_shape),
        )

    def __call__(self, z):
        x_reconstructed = self._layer(z)
        #print(x.size())
        #x_reconstructed = x.view(-1, *self._output_shape)
        return torch.sigmoid(x_reconstructed)


class VAE(torch.nn.Module):
    def __init__(self, image_shape, hid_size=512, latent_size=10):
        super().__init__()
        self._encoder = Encoder(
            image_shape, hid_size=hid_size, latent_size=latent_size)
        self._decoder = Decoder(
            image_shape, hid_size=hid_size, latent_size=latent_size)

    def forward(self, x):
        z, (mean, stddev) = self.encode(x)
        logits, mean_image, sampled_image = self.decode(z)
        return mean_image, sampled_image, logits, z, mean, stddev

    def encode(self, x):
        mean, stddev = self._encoder(x)
        normal_sample = torch.randn_like(stddev)

        z = mean + normal_sample * stddev
        return z, (mean, stddev)

    def decode(self, z):
        logits = self._decoder(z)
        mean_image = torch.sigmoid(logits)
        sampled_image = torch.bernoulli(mean_image)
        return logits, mean_image, sampled_image


def kl_gaussian(mean, var):
    kl_divergence_vector = 0.5 * (-torch.log(var) - 1.0 + var + mean**2)
    return torch.sum(kl_divergence_vector, axis=-1)


def binary_cross_entropy(logits, x):
    x = x.view(x.shape[0], -1)
    logits = logits.view(x.shape[0], -1)
    return -torch.sum(x * logits - torch.log(1 + torch.exp(logits)))


class ELBO(torch.nn.Module):
    """
        Calculate the ELBO.
        mean: mean of the q(z|x).
        stddev: stddev of the q(z|x).
    """

    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        _, _, logits, _, mean, stddev = y_pred
        log_likelihood = -torch.nn.functional.binary_cross_entropy(
            logits, torch.sigmoid(y_true), reduction=self.reduction)
        kl = kl_gaussian(mean, stddev**2)
        elbo = torch.mean(log_likelihood - kl)
        return -elbo