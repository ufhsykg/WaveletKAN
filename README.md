# WaveletKAN

This is a lightweight implementation of a PyTorch layer using wavelet transformations. It's inspired by the [FourierKAN](https://github.com/GistNoesis/FourierKAN) project.

## Performance

Compared with `FourierKANLayer`, `WaveletKANLayer` has a significant improvement in computational speed, and when we replaced the two in several tests, Wavelet's performance was slightly higher than Fourier's

## Inspiration

This project draws on concepts from the FourierKAN, adapting them by utilizing wavelet transforms instead of Fourier transforms. Specifically, we leverage the sigmoid function within the wavelet base calculation (`wavelet_base = torch.sigmoid(w * xrshp)`) to speed up the wavelet transform operations. There may be better choices for sigmoid

## Installation

Assuming you have Python and PyTorch set up, you can integrate this layer into your existing projects by including the `WaveletKANLayer` class.

## License

This project is made available under the MIT License.
