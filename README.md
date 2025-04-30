# Driving_Simulator
A PyTorch based training framework for a Machine Learning Driving Simulator like the GAIA-1 [4]. Currently only the image tokenizer is implemented. A transformer-based, action-conditioned world model will follow soon.

<img src="img/VQ_GAN.png" alt="Architecture">

The gifs below show on the top the (256,512) original video and on the bottom a reconstruction from 16 x 32 (spatial compression rate 16) visual tokens from a learned codebook consisting of 1024 vectors of size 256 each.

![Original video](img/original.gif)
![Reconstructed video](img/reconstructed.gif)

## Getting started

### Setup conda environment

```console
$ conda env create -f environment.yml
```

### Train the model

Run the following commands in the terminal:

```console
$ conda activate driving_simulator
$ cd tokenizer
$ python cli.py train --image_root_path <folder-path>
```

To see all possible arguments for the training command run

```console
$ python cli.py train -h
```

### Monitor a training run on a server

1. On the **server**:

```console
tensorboard --logdir <logs-path> --port 6006
```

2. Forward the port on the **client** via SSH:

```console
 ssh -L 16006:localhost:6006 <user>@<server-ip>
```

Now you should be able to open the *TensorBoard* dashboard in the browser at `http://127.0.0.1:16006`.

## Implemented features

### Tokenizer

* Finite Scalar Quantization (FSQ) [5]
* GAN loss [2]
* LPIPS perceptual loss [3]
* Entropy calculation for codebook utilization monitoring during training

## References

[1] van den Oord, A., Li, Y., & Vinyals, O. (2017). "Neural discrete representation learning".

[2] Esser, P., Rombach, R., & Ommer, B. (2021). "Taming Transformers for High-Resolution Image Synthesis".

[3] Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric".

[4] Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J., & Corrado, G. (2023). "GAIA-1: A Generative World Model for Autonomous Driving".

[5] Mentzer, F., Minnen, D., Agustsson, E., Tschannen, M. (2023). "Finite Scalar Quantization: VQ-VAE Made Simple".
