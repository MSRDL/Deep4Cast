import numpy as np
import torch


class ConcreteDropout(torch.nn.Module):
    """Applies Dropout to the input, even at prediction time and learns dropout probability
    from the data.
    
    In convolutional neural networks, we can use dropout to drop entire channels using
    the 'channel_wise' argument.
    
    Arguments:
        * dropout_regularizer (float): Should  be set to 2 / N, where N is the number of training examples.
        * init_range (tuple): Initial range for dropout probabilities.
        * channel_wise (boolean): apply dropout over all input or across convolutional channels.

    """
    def __init__(self,
                 dropout_regularizer=1e-5,
                 init_range=(0.1, 0.3),
                 channel_wise=False):
        super(ConcreteDropout, self).__init__()
        self.dropout_regularizer = dropout_regularizer
        self.init_range = init_range
        self.channel_wise = channel_wise

        # Initialize dropout probability
        init_min = np.log(init_range[0]) - np.log(1. - init_range[0])
        init_max = np.log(init_range[1]) - np.log(1. - init_range[1])
        self.p_logit = torch.nn.Parameter(
            torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x):
        """Returns input but with randomly dropped out values."""
        # Get the dropout probability
        p = torch.sigmoid(self.p_logit)

        # Apply Concrete Dropout to input
        out = self._concrete_dropout(x, p)

        # Regularization term for dropout parameters
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        # The size of the dropout regularization depends on the kind of input
        if self.channel_wise:
            # Dropout only applied to channel dimension
            input_dim = x.shape[1]
        else:
            # Dropout applied to all dimensions
            input_dim = np.prod(x.shape[1:])
        dropout_regularizer *= self.dropout_regularizer * input_dim

        return out, dropout_regularizer.mean()

    def _concrete_dropout(self, x, p):
        # Empirical parameters for the concrete distribution
        eps = 1e-7
        temp = 0.1

        # Apply Concrete dropout channel wise or across all input
        if self.channel_wise:
            unif_noise = torch.rand_like(x[:, :, [0]])
        else:
            unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob

        # Need to make sure we have the right shape for the Dropout mask
        if self.channel_wise:
            random_tensor = random_tensor.repeat([1, 1, x.shape[2]])

        # Drop weights
        retain_prob = 1 - p
        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

