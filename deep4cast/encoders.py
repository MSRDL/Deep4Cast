import numpy as np
import torch

from deep4cast import custom_layers


class WaveNet(torch.nn.Module):
    """Implements `WaveNet` architecture for time series forecasting. Inherits 
    from pytorch `Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_.
    Vector forecasts are made via a fully-connected layer.

    References:
        - `WaveNet: A Generative Model for Raw Audio <https://arxiv.org/pdf/1609.03499.pdf>`_
    
    Arguments:
        * input_channels (int): Number of covariates in input time series.
        * hidden_channels (int): Number of channels in convolutional hidden layers.
        * skip_channels (int): Number of channels in convolutional layers for skip connections.
        * n_layers (int): Number of layers per Wavenet block (determines receptive field size).
        * dilation (int): Dilation factor for temporal convolution.

    """
    def __init__(self,
                 input_channels,
                 hidden_channels=64,
                 skip_channels=64,
                 n_layers=7,
                 n_blocks=1,
                 dilation=2):
        """Inititalize variables."""
        super(WaveNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.n_layers = n_layers
        self.dilation = dilation
        self.dilations = [dilation**i for i in range(n_layers)]

        # Set up first layer for input
        self.dout_conv_input = custom_layers.ConcreteDropout(channel_wise=True)
        self.conv_input = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1
        )

        # Set up main WaveNet layers
        self.dout, self.conv, self.skip, self.resi = [], [], [], []
        for d in self.dilations:
            self.dout.append(custom_layers.ConcreteDropout(channel_wise=True))
            self.conv.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                             out_channels=hidden_channels,
                                             kernel_size=2,
                                             dilation=d))
            self.skip.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                             out_channels=skip_channels,
                                             kernel_size=1))
            self.resi.append(torch.nn.Conv1d(in_channels=hidden_channels,
                                             out_channels=hidden_channels,
                                             kernel_size=1))
        self.dout = torch.nn.ModuleList(self.dout)
        self.conv = torch.nn.ModuleList(self.conv)
        self.skip = torch.nn.ModuleList(self.skip)
        self.resi = torch.nn.ModuleList(self.resi)

        # Set up nonlinear output layers
        self.dout_conv_post = custom_layers.ConcreteDropout(channel_wise=True)
        self.conv_post = torch.nn.Conv1d(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=1
        )

    def forward(self, inputs):
        """Returns embedding vectors.
        
        Arguments:
            * inputs (``torch.Tensor``): time series input to make forecasts for

        """
        # Input layer
        output, res_conv_input = self.dout_conv_input(inputs)
        output = self.conv_input(output)

        # Loop over WaveNet layers and blocks
        regs, skip_connections = [], []
        for dout, conv, skip, resi in zip(self.dout, self.conv, self.skip, self.resi):
            layer_in = output
            output, reg = dout(layer_in)
            output = conv(output)
            output = torch.nn.functional.relu(output)
            skip = skip(output)
            output = resi(output)
            output = output + layer_in[:, :, -output.size(2):]
            regs.append(reg)
            skip_connections.append(skip)

        # Sum up regularizer terms and skip connections
        regs = sum(r for r in regs)
        output = sum([s[:, :, -output.size(2):] for s in skip_connections])

        # Nonlinear output layers
        output, res_conv_post = self.dout_conv_post(output)
        output = torch.nn.functional.relu(output)
        output = self.conv_post(output)
        output = torch.nn.functional.relu(output)
        output = output[:, :, [-1]]
        output = output.transpose(1, 2)

        # Regularization terms
        regularizer = res_conv_input \
            + regs \
            + res_conv_post

        return output, regularizer

    @property
    def n_parameters(self):
        """Returns the number of model parameters."""
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    @property
    def receptive_field_size(self):
        """Returns the length of the receptive field."""
        return self.dilation * max(self.dilations)

