import torch
import numpy as np

from deep4cast import custom_layers


class Linear(torch.nn.Module):
    r"""Implements a simple one-layer linear decoder network with Dropout
    
    Arguments:
        * input_dim (int): Number of input nodes
        * output_params (int): Number of loss parameters.
        * output_channels (int): Number of target time series.
        * horizon (int): Number of time steps to forecast.

    """
    def __init__(self,
                 input_dim,
                 output_params,
                 output_channels,
                 horizon):
        """Inititalize variables."""
        super(Linear, self).__init__()
        self.input_dim = output_channels
        self.output_params = output_params
        self.output_channels = output_channels
        self.horizon = horizon

        # Set up layers
        self.dout, self.linear = [], []
        for __ in self.output_params:
            self.dout.append(custom_layers.ConcreteDropout())
            self.linear.append(torch.nn.Linear(self.input_dim, self.horizon*self.output_channels))
        self.dout = torch.nn.ModuleList(self.dout)
        self.linear = torch.nn.ModuleList(self.linear)

    def forward(self, inputs):
        """Returns forecasts based on embedding vectors.
        
        Arguments:
            * inputs (``torch.Tensor``): embedding vectors to generate forecasts for

        """
        batch_size = inputs.shape[0]

        # Apply dropout before linear layers
        outputs, regularizer = [], []
        for dout, linear in zip(self.dout, self.linear):
            # Need to apply dropout first and collect regularizer terms
            output_dout, reg_dout = dout(inputs)
            output = linear(output_dout)

            # Reshape everything to match desired output horizon
            output = output.reshape((batch_size, self.output_channels, self.horizon))
            outputs.append(output)
            regularizer.append(reg_dout)
        outputs = torch.nn.ModuleList(outputs)
        regularizer = torch.nn.ModuleList(regularizer)

        # Regularization terms need to be added up
        regularizer = regularizer.sum()

        return {'outputs': outputs, 'regularizer': regularizer}

    @property
    def n_parameters(self):
        """Returns the number of model parameters."""
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

