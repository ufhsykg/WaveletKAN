# WaveletKAN

This is a lightweight implementation of a PyTorch layer using wavelet transformations. It's inspired by the [FourierKAN](https://github.com/GistNoesis/FourierKAN) project.

## Performance

Compared with `FourierKANLayer`, `WaveletKANLayer` has a significant improvement in computational speed, and when we replaced the two in several tests, Wavelet's performance was slightly higher than Fourier's

```
Test results on MNIST

Train Epoch: 1 [0/60000 (0%)]   Loss: 4.771067
Train Epoch: 1 [6400/60000 (11%)]       Loss: 0.487188
Train Epoch: 1 [12800/60000 (21%)]      Loss: 0.230800
Train Epoch: 1 [19200/60000 (32%)]      Loss: 0.185986
Train Epoch: 1 [25600/60000 (43%)]      Loss: 0.340212
Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.223590
Train Epoch: 1 [38400/60000 (64%)]      Loss: 0.123554
Train Epoch: 1 [44800/60000 (75%)]      Loss: 0.269612
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.117765
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.301055

Test set: Average loss: 0.0023, Accuracy: 9553/10000 (96%)
```

## Inspiration

This project draws on concepts from the FourierKAN, adapting them by utilizing wavelet transforms instead of Fourier transforms. Specifically, we leverage the sigmoid function within the wavelet base calculation (`wavelet_base = torch.sigmoid(w * xrshp)`) to speed up the wavelet transform operations. There may be better choices for sigmoid

## Installation

Assuming you have Python and PyTorch set up, you can integrate this layer into your existing projects by including the `WaveletKANLayer` class.

## License

This project is made available under the MIT License.
