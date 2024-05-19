import torch
import torch.nn as nn
import numpy as np


class WaveletKANLayer(torch.nn.Module):
    def __init__(self, inputdim, outdim, wavelet_size, addbias=True):
        super(WaveletKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.wavelet_size = wavelet_size  # Wavelet size

        # Wavelet coefficients initialization
        self.waveletcoeffs = torch.nn.Parameter(
            torch.randn(outdim, inputdim, wavelet_size)
            / (np.sqrt(inputdim) * np.sqrt(wavelet_size))
        )

        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        # Wavelet part
        w = torch.reshape(
            torch.arange(1, self.wavelet_size + 1, device=x.device),
            (1, 1, 1, self.wavelet_size),
        )
        wavelet_base = torch.sigmoid(w * xrshp)

        y_wavelet = torch.einsum("boiw,oik->bo", wavelet_base, self.waveletcoeffs)

        y = y_wavelet  # Test

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y


if __name__ == "__main__":
    input_dim = 10
    output_dim = 5
    wavelet_size = 3
    model = WaveletKANLayer(input_dim, output_dim, wavelet_size)
    batch_size = 2
    x = torch.randn(batch_size, input_dim)

    output = model(x)

    print("Output of the WaveletKANLayer:")
    print(output)
