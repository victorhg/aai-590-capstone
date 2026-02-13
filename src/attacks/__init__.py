# Initialize the attacks module
from .pgd import PGDAttack
from .uap import UniversalPerturbation

__all__ = ['PGDAttack', 'UniversalPerturbation']
