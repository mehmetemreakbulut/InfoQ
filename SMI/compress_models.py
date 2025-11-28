import torch
class CIFAR10_ConvEncoder(torch.nn.Module):
    """
    CIFAR10 Convolutional encoder.

    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Noise.
        self.dropout = torch.nn.Dropout(0.1)

        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Tanh()

        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding='same')

        self.maxpool2d = torch.nn.MaxPool2d((2,2))

        # Dense.
        self.linear_1 = torch.nn.Linear(64*4*4, 128)
        self.linear_2 = torch.nn.Linear(128, self.latent_dim)


    def forward(self, x: torch.tensor) -> torch.tensor:
        # Convolution №1
        x = self.dropout(x)
        x = self.conv2d_1(x)
        x = self.maxpool2d(x)
        layer_1 = self.activation(x)

        # Convolution №2
        x = self.dropout(layer_1)
        x = self.conv2d_2(x)
        x = self.maxpool2d(x)
        layer_2 = self.activation(x)

        # Convolution №3
        x = self.dropout(layer_2)
        x = self.conv2d_3(x)
        x = self.maxpool2d(x)
        layer_3 = self.activation(x)

        # Dense №1
        x = torch.flatten(layer_3, 1)
        x = self.linear_1(x)
        layer_4 = self.activation(x)

        # Dense №2
        x = self.linear_2(layer_4)
        layer_5 = self.sigmoid(x)

        return layer_5

class AdditiveGaussianNoise(torch.nn.Module):
    """
    Additive Gaussian noise.
    """

    def __init__(self, sigma : float=0.1, relative_scale : bool=True, enabled_on_inference : bool=False):
        """
        Initialization.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of additive Gaussian noise.
        relative_scale : bool, optional
            Scale the noise to preserve noise-to-signal ratio.
        enabled_on_inference : bool, optional
            Inject noise on inference.
        """

        super().__init__()
        self.sigma = sigma
        self.relative_scale = relative_scale
        self.enabled_on_inference = enabled_on_inference

        # A separate parameter for noise.
        # Needed in order to generate noise on the desired device.
        #self.noise = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.noise.requires_grad_(False)
        self.register_buffer('noise', torch.tensor(0.0, dtype=torch.float32))  # Is preferable in case of multiple devices.


    def forward(self, x):
        if (self.training or self.enabled_on_inference) and self.sigma != 0:
            scale = self.sigma * x if self.relative_scale else self.sigma
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class CIFAR10_ConvDecoder(torch.nn.Module):
    """
    CIFAR10 Convolutional decoder.

    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(64, 32, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2d_2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2d_3 = torch.nn.Conv2d(32, 16, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2d_4 = torch.nn.Conv2d(16, 3, kernel_size=3, padding='same', padding_mode='reflect')

        self.upsample = torch.nn.Upsample(scale_factor=2)

        # Dense.
        self.linear_1 = torch.nn.Linear(latent_dim, 128)
        self.linear_2 = torch.nn.Linear(128, 64*4*4)


    def forward(self, x: torch.tensor) -> torch.tensor:
        # Dense №1
        x = self.linear_1(x)
        layer_1 = self.activation(x)

        # Dense №2
        x = self.linear_2(layer_1)
        layer_2 = self.activation(x)

        # Convolution №1
        x = torch.reshape(layer_2, (-1, 64, 4, 4))
        x = self.conv2d_1(x)
        x = self.upsample(x)
        layer_3 = self.activation(x)

        # Convolution №2
        x = self.conv2d_2(layer_3)
        x = self.upsample(x)
        layer_4 = self.activation(x)

        # Convolution №3
        x = self.conv2d_3(layer_4)
        x = self.upsample(x)
        layer_5 = self.activation(x)

        # Convolution №4
        x = self.conv2d_4(layer_5)
        layer_6 = x

        return layer_6
    
class Autoencoder(torch.nn.Module):
    """
    Autoencoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder.
    decoder : torch.nn.Module
        Decoder.
    latent_dim : int
        Latent representation dimension.
    sigma : float
        Standard deviation of additive Gaussian noise,
        injected into the latent representation.
    """

    def __init__(self, encoder, decoder, sigma: float=0.1):
        super().__init__()
        #self.sigma = sigma

        # Encoder and decoder.
        self.encoder = encoder
        self.decoder = decoder

        # Noise.
        self.agn = AdditiveGaussianNoise(sigma=sigma, enabled_on_inference=False)


    def forward(self, x: torch.tensor) -> torch.tensor:
        latent = self.encoder(x)
        latent = self.agn(latent)

        return self.decoder(latent)


    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)


    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)

def get_outputs(model, dataloader, device) -> list():
    """
    Get outputs of the model.
    """

    # Exit training mode.
    was_in_training = model.training
    model.eval()

    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch['image'], batch['label']

            outputs.append(model(x.to(device)).detach().cpu())

    outputs = torch.cat(outputs)

    # Return to the original mode.
    model.train(was_in_training)

    return outputs