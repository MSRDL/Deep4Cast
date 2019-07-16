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
        * output_params (int): Number of loss parameters.
        * output_channels (int): Number of target time series.
        * horizon (int): Number of time steps to forecast.
        * hidden_channels (int): Number of channels in convolutional hidden layers.
        * skip_channels (int): Number of channels in convolutional layers for skip connections.
        * n_layers (int): Number of layers per Wavenet block (determines receptive field size).
        * dilation (int): Dilation factor for temporal convolution.
    """
    def __init__(self,
                 input_channels,
                 output_params,
                 output_channels,
                 horizon,
                 hidden_channels=64,
                 skip_channels=64,
                 n_layers=7,
                 dilation=2):
        """Inititalize variables."""
        super(WaveNet, self).__init__()
        self.input_channels = input_channels
        self.output_params = output_params
        self.output_channels = output_channels
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.n_layers = n_layers
        self.dilation = dilation
        self.dilations = [dilation**i for i in range(n_layers)]

        # Set up first layer for input
        self.dout_conv_input = custom_layers.ConcreteDropout(channel_wise=True)
        self.conv_input = torch.nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=hidden_channels,
            kernel_size=1
        )

        # Set up main WaveNet layers
        self.edout, self.conv, self.skip, self.resi = [], [], [], []
        for d in self.dilations:
            self.edout.append(custom_layers.ConcreteDropout(channel_wise=True))
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
        self.edout = torch.nn.ModuleList(self.edout)
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

        # Set up decoder layers
        self.ddout, self.linear = [], []
        for __ in range(self.output_params):
            self.ddout.append(custom_layers.ConcreteDropout())
            self.linear.append(torch.nn.Linear(self.skip_channels, self.horizon*self.output_channels))
        self.ddout = torch.nn.ModuleList(self.ddout)
        self.linear = torch.nn.ModuleList(self.linear)

    def encode(self, inputs):
        """Returns embedding vectors.
        
        Arguments:
            * inputs (``torch.Tensor``): time series input to make forecasts for
        """
        # Input layer
        output, res_conv_input = self.dout_conv_input(inputs)
        output = self.conv_input(output)

        # Loop over WaveNet layers and blocks
        regs, skip_connections = [], []
        for edout, conv, skip, resi in zip(self.edout, self.conv, self.skip, self.resi):
            layer_in = output
            output, reg = edout(layer_in)
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

    def decode(self, inputs):
        """Returns forecasts based on embedding vectors.
        
        Arguments:
            * inputs (``torch.Tensor``): embedding vectors to generate forecasts for

        """
        batch_size = inputs.shape[0]

        # Apply dropout before linear layers
        outputs, regularizer = [], []
        for ddout, linear in zip(self.ddout, self.linear):
            # Need to apply dropout first and collect regularizer terms
            output_ddout, reg_ddout = ddout(inputs)
            output = linear(output_ddout)

            # Reshape everything to match desired output horizon
            output = output.reshape((batch_size, self.output_channels, self.horizon))
            outputs.append(output)
            regularizer.append(reg_ddout)

        # Regularization terms need to be added up
        regularizer = sum(regularizer)

        return outputs, regularizer
    
    def forward(self, inputs):
        """Forward function."""
        outputs, reg_encoder = self.encode(inputs)
        outputs, reg_decoder = self.decode(outputs)

        # Regularization
        regularizer = reg_encoder + reg_decoder

        return outputs, regularizer

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
