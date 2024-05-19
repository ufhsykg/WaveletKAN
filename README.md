# WaveletKAN

This is a lightweight implementation of a PyTorch layer using wavelet transformations. It's inspired by the [FourierKAN](https://github.com/GistNoesis/FourierKAN) project.

## Performance

In our testing, when substituting traditional MLP layers with KAN layers in a similar MLP-style architecture, the `WaveletKANLayer` not only performed slightly better than the `FourierKANLayer` in terms of accuracy and learning efficiency but also showed a significant improvement in computational speed. 

## Inspiration

This project draws on concepts from the FourierKAN, adapting them by utilizing wavelet transforms instead of Fourier transforms. Specifically, we leverage the sigmoid function within the wavelet base calculation (`wavelet_base = torch.sigmoid(w * xrshp)`) to speed up the wavelet transform operations. There may be better choices for sigmoid

## Installation

Assuming you have Python and PyTorch set up, you can integrate this layer into your existing projects by including the `WaveletKANLayer` class.

## License

This project is made available under the MIT License.
