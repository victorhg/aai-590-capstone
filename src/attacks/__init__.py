# Initialize the attacks module
from .pgd import PGDAttack
from .uap import UniversalPerturbation, UAPDataset


__all__ = ['PGDAttack', 
           'UniversalPerturbation',
           'UAPDataset']
