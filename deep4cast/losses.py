import torch
import numpy as np

from abc import ABC, abstractmethod


class Loss(ABC):    
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError




