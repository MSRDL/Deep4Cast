import numpy as np
import torch

from deep4cast import custom_layers


class WaveNet(torch.nn.Module):
    """Implements `WaveNet` architecture for time series forecasting. Inherits 
    from pytorch `Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_.
    Vector forecasts are made via a fully-connected linear layer.

    References:
        - `WaveNet: A Generative Model for Raw Audio <https://arxiv.org/pdf/1609.03499.pdf>`_
    
    Arguments:
        * input_channels (int): Number of covariates in input time series.
        * output_channels (int): Number of target time series.
        * horizon (int): Number of time steps to forecast.
        * hidden_channels (int): Number of channels in convolutional hidden layers.
        * skip_channels (int): Number of channels in convolutional layers for skip connections.
        * n_layers (int): Number of layers per Wavenet block (determines receptive field size).
        * n_blocks (int): Number of Wavenet blocks.
        * dilation (int): Dilation factor for temporal convolution.

    """
    def __init__(self,
                 input_channels,
                 output_channels,
                 horizon,
                 hidden_channels=64,
                 skip_channels=64,
                 n_layers=7,
                 n_blocks=1,
                 dilation=2):
        """Inititalize variables."""
        super(WaveNet, self).__init__()
        self.output_channels = output_channels
        self.horizon = horizon
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.dilation = dilation
        self.dilations = [dilation**i for i in range(n_layers)] * n_blocks

        # Set up first layer for input
        self.do_conv_input = custom_layers.ConcreteDropout(channel_wise=True)
        self.conv_input = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1
        )

        # Set up main WaveNet layers
        self.do, self.conv, self.skip, self.resi = [], [], [], []
        for d in self.dilations:
            self.do.append(custom_layers.ConcreteDropout(channel_wise=True))
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
        self.do = torch.nn.ModuleList(self.do)
        self.conv = torch.nn.ModuleList(self.conv)
        self.skip = torch.nn.ModuleList(self.skip)
        self.resi = torch.nn.ModuleList(self.resi)

        # Set up nonlinear output layers
        self.do_conv_post = custom_layers.ConcreteDropout(channel_wise=True)
        self.conv_post = torch.nn.Conv1d(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=1
        )
        self.do_linear_mean = custom_layers.ConcreteDropout()
        self.do_linear_std = custom_layers.ConcreteDropout()
        self.do_linear_df = custom_layers.ConcreteDropout()
        self.linear_mean = torch.nn.Linear(
            skip_channels, horizon*output_channels)
        self.linear_std = torch.nn.Linear(
            skip_channels, horizon*output_channels)
        self.linear_df = torch.nn.Linear(
            skip_channels, horizon*output_channels)

    def forward(self, inputs):
        """Forward function."""
        output, reg_e = self.encode(inputs)
        output_mean, output_std, output_df, reg_d = self.decode(output)

        # Regularization
        regularizer = reg_e + reg_d

        return {'df': output_df, 'loc': output_mean, 'scale': output_std, 'regularizer': regularizer}

    def encode(self, inputs: torch.Tensor):
        """Returns embedding vectors.
        
        Arguments:
            * inputs: time series input to make forecasts for

        """
        # Input layer
        output, res_conv_input = self.do_conv_input(inputs)
        output = self.conv_input(output)

        # Loop over WaveNet layers and blocks
        regs, skip_connections = [], []
        for do, conv, skip, resi in zip(self.do, self.conv, self.skip, self.resi):
            layer_in = output
            output, reg = do(layer_in)
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
        output, res_conv_post = self.do_conv_post(output)
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

    def decode(self, inputs: torch.Tensor):
        """Returns forecasts based on embedding vectors.
        
        Arguments:
            * inputs: embedding vectors to generate forecasts for

        """
        # Apply dense layer to match output length
        output_mean, res_linear_mean = self.do_linear_mean(inputs)
        output_std, res_linear_std = self.do_linear_std(inputs)
        output_df, res_linear_df = self.do_linear_df(inputs)
        output_mean = self.linear_mean(output_mean)
        output_std = self.linear_std(output_std).exp()
        output_df = self.linear_df(output_df).exp()

        # Reshape the layer output to match targets
        # Shape is (batch_size, output_channels, horizon)
        batch_size = inputs.shape[0]
        output_mean = output_mean.reshape(
            (batch_size, self.output_channels, self.horizon)
        )
        output_std = output_std.reshape(
            (batch_size, self.output_channels, self.horizon)
        )
        output_df = output_df.reshape(
            (batch_size, self.output_channels, self.horizon)
        )

        # Regularization terms
        regularizer = res_linear_mean + res_linear_std + res_linear_df

        return output_mean, output_std, output_df, regularizer

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
