from inspect import getfullargspec
import time
from typing import Union

import numpy as np
import torch


class Forecaster():
    """Handles training of a PyTorch model. Can be used to generate samples
    from approximate posterior predictive distribution.

    :param model: PyTorch neural network model :ref:`models`
    :param loss: PyTorch distribution
    :param optimizer: PyTorch optimizer
    :param n_epochs: number of training epochs
    :param device: device used for training (cpu or cuda)
    :param checkpoint_path: path for writing model checkpoints
    :param verbose: switch to toggle verbosity of forecaster during fitting
    :param nan_budget: how many time the forecaster will try to continue batcvh
        training when NaN encountered.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss=torch.distributions.Normal,
                 optimizer=torch.optim.Adam,
                 n_epochs=1,
                 device='cpu',
                 checkpoint_path='./',
                 verbose=True,
                 nan_budget=5):
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        self.model = model.to(device)
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.loss = loss
        self.history = {'training': [], 'validation': []}
        self.checkpoint_path = checkpoint_path
        self.nan_budget = nan_budget
        self.verbose = verbose

    def fit(self, dataloader_train, dataloader_val=None, eval_model=False):
        """Fit model to data."""
        if self.verbose:
            print('Number of model parameters being fitted: {}.'.format(self.model.n_parameters))

        # Iterate over training epochs
        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            self.train(dataloader_train, epoch, start_time)
            self.save_checkpoint()            
            if eval_model:
                train_loss = self.evaluate(dataloader_train)
                if self.verbose: print('\nTraining error: {:1.2e}.'.format(train_loss))
                self.history['training'].append(train_loss) 
            if dataloader_val:
                val_loss = self.evaluate(dataloader_val)
                if self.verbose: print('Validation error: {:1.2e}\n.'.format(val_loss))
                self.history['validation'].append(val_loss)

    def train(self, dataloader, epoch, start_time):
        """Perform training for one epoch."""
        n_trained = 0
        nan_budget = self.nan_budget
        for idx, batch in enumerate(dataloader):
            # Send batch to device
            inputs = batch['X'].to(self.device)
            targets = batch['y'].to(self.device)

            # Backpropagation
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            reg = outputs.pop('regularizer')
            loss = -self.loss(**outputs).log_prob(targets).mean() + reg
            
            # We give the forecaster a chance to get out of NaNs using a budget
            if torch.isnan(loss):
                nan_budget -= 1
                if not nan_budget:
                    raise ValueError('NaN in training loss. NaN budget depleted.')
                else:
                    continue
            else:
                nan_budget = self.nan_budget
            loss.backward()
            self.optimizer.step()

            # Status update for the user
            if self.verbose:
                n_trained += len(inputs)
                n_total = len(dataloader.dataset)
                percentage = 100.0 * idx / len(dataloader)
                elapsed = time.time() - start_time
                remaining = elapsed*((self.n_epochs*n_total)/((epoch-1)*n_total + n_trained) - 1)
                status = '\rEpoch {}/{} [{}/{} ({:.0f}%)]\t' \
                       + 'Loss: {:.6f}\t' \
                       + 'Elapsed/Remaining: {:.0f}m{:.0f}s/{:.0f}m{:.0f}s   '
                print(
                    status.format(
                        epoch, 
                        self.n_epochs, 
                        n_trained, 
                        n_total, 
                        percentage,
                        loss.item(),
                        elapsed // 60,
                        elapsed % 60,
                        remaining // 60, 
                        remaining % 60
                    ), 
                    end=""
                )            

    def evaluate(self, dataloader, n_samples=1):
        """Evaluate a model on a dataset.
        
        Returns the approximate min negative log likelihood of the model averaged over dataset

        """
        max_llikelihood = [0]*n_samples
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                
                # Forward pass through the model
                outputs = self.model(inputs)
                outputs.pop('regularizer')
                
                # Calculate loss (typically probability density)    
                for i in range(n_samples):
                    loss = self.loss(**outputs).log_prob(targets) 
                    max_llikelihood[i] += loss.sum().item()
            max_llikelihood = np.max(max_llikelihood)
            
        return -max_llikelihood / len(dataloader.dataset)

    def predict(self, dataloader, n_samples=100):
        """Generate predictions."""
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                inputs = batch['X'].to(self.device)
                samples = []
                for i in range(n_samples):
                    outputs = self.model(inputs)
                    outputs.pop('regularizer')
                    samples.append(self.loss(**outputs).sample((1,)).cpu().numpy())
                samples = np.concatenate(samples, axis=0)
                predictions.append(samples)
            predictions = np.concatenate(predictions, axis=1)

        return predictions

    def save_checkpoint(self):
        """Save a complete PyTorch model checkpoint."""
        filename = self.checkpoint_path
        filename += 'checkpoint_model.pt'
        save_dict = {}
        save_dict['model_def'] = self.model
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        save_dict['loss'] = self.loss
        torch.save(save_dict, filename)

