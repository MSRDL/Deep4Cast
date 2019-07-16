import copy
import time

import numpy as np
import torch


class Forecaster():
    """Handles training of a PyTorch model and can be used to generate samples
    from approximate posterior predictive distribution.

    Arguments:
        * model (``torch.nn.Module``): Instance of Deep4cast :class:`models`.
        * loss (``torch.distributions``): Instance of PyTorch `distribution <https://pytorch.org/docs/stable/distributions.html>`_.
        * optimizer (``torch.optim``): Instance of PyTorch `optimizer <https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer>`_.
        * n_epochs (int): Number of training epochs.
        * device (str): Device used for training (`cpu` or `cuda`).
        * checkpoint_path (str): File system path for writing model checkpoints.
        * verbose (bool): Verbosity of forecaster.
    
    """
    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 n_epochs=1,
                 device='cpu',
                 checkpoint_path='./',
                 verbose=True):
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        self.model = model.to(device)
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.loss = loss
        self.history = {'training': [], 'validation': []}
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

    def fit(self,
            dataloader_train,
            dataloader_val=None,
            eval_model=False):
        """Fits a model to a given a dataset.
        
        Arguments:
            * dataloader_train (``torch.utils.data.DataLoader``): Training data.
            * dataloader_val (``torch.utils.data.DataLoader``): Validation data.
            * eval_model (bool): Flag to switch on model evaluation after every epoch.
        
        """
        # Iterate over training epochs
        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            self._train(dataloader_train, epoch, start_time)
            self._save_checkpoint()            
            if eval_model:
                train_loss = self._evaluate(dataloader_train)
                if self.verbose: print('\nTraining error: {:1.2e}.'.format(train_loss))
                self.history['training'].append(train_loss) 
            if dataloader_val:
                val_loss = self._evaluate(dataloader_val)
                if self.verbose: print('Validation error: {:1.2e}\n.'.format(val_loss))
                self.history['validation'].append(val_loss)

    def _train(self, dataloader, epoch, start_time):
        """Perform training for one epoch.

        Arguments:
            * dataloader (``torch.utils.data.DataLoader``): Training data.
            * epoch (int): Current training epoch.
            * start_time (``time.time``): Clock time of training start.
            
        """
        n_trained = 0
        for idx, batch in enumerate(dataloader):
            # Send batch to device
            inputs = batch['X'].to(self.device)
            targets = batch['y'].to(self.device)

            # Backpropagation
            self.optimizer.zero_grad()
            outputs, reg = self.model(inputs)
            loss = self.loss.evaluate(outputs, targets) + reg
            if torch.isnan(loss.mean()):
                raise ValueError('NaN in training loss.')
            loss.backward()
            self.optimizer.step()

            # Status update for the user
            if self.verbose:
                n_trained += len(inputs)
                n_total = len(dataloader.dataset)
                percentage = 100.0 * (idx + 1) / len(dataloader)
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
                        loss.mean().item(),
                        elapsed // 60,
                        elapsed % 60,
                        remaining // 60, 
                        remaining % 60
                    ), 
                    end=""
                )            

    def _evaluate(self, dataloader, n_samples=10):
        """Returns the approximate min negative log likelihood of the model 
        averaged over dataset.

        Arguments:
            * dataloader (``torch.utils.data.DataLoader``): Evaluation data.
            * n_samples (int): Number of forecast samples.
        
        """
        max_llikelihood = [0]*n_samples
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                
                # Forward pass through the model
                outputs, __ = self.model(inputs)
                # Calculate loss (typically probability density)    
                for i in range(n_samples):
                    loss = self.loss.evaluate(outputs, targets) 
                    max_llikelihood[i] += loss.item()
            max_llikelihood = np.max(max_llikelihood)
            
        return -max_llikelihood / len(dataloader.dataset)

    def predict(self, dataloader, n_samples=100) -> np.array:
        """Generates predictions.

        Arguments:
            * dataloader (``torch.utils.data.DataLoader``): Data to make forecasts.
            * n_samples (int): Number of forecast samples.
        
        """
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                inputs = batch['X'].to(self.device)
                samples = []
                for i in range(n_samples):
                    outputs = self.model(inputs)
                    outputs.pop('regularizer')
                    outputs = self.loss.sample()
                    batch['y'] = outputs[0]
                    outputs = copy.deepcopy(batch)
                    outputs = dataloader.dataset.transform.untransform(outputs)
                    samples.append(outputs['y'][None, :])
                samples = np.concatenate(samples, axis=0)
                predictions.append(samples)
            predictions = np.concatenate(predictions, axis=1)

        return predictions

    def embed(self, dataloader, n_samples=100) -> np.array:
        """Generate embedding vectors.

        Arguments:
            * dataloader (``torch.utils.data.DataLoader``): Data to make embedding vectors.
            * n_samples (int): Number of forecast samples.
        
        """
        with torch.no_grad():
            predictions = []
            for batch in dataloader:
                inputs = batch['X'].to(self.device)
                samples = []
                for i in range(n_samples):
                    outputs, __ = self.model.encode(inputs)
                    samples.append(outputs.cpu().numpy())
                samples = np.array(samples)
                predictions.append(samples)
            predictions = np.concatenate(predictions, axis=1)

        return predictions
    
    def _save_checkpoint(self):
        """Save a complete PyTorch model checkpoint."""
        filename = self.checkpoint_path
        filename += 'checkpoint_model.pt'
        save_dict = {}
        save_dict['model_def'] = self.model
        save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        save_dict['loss'] = self.loss
        torch.save(save_dict, filename)

