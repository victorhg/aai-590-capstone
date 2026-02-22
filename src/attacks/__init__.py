# Initialize the attacks module
from .pgd import PGDAttack
from .uap import UniversalPerturbation, UAPDataset


__all__ = ['PGDAttack', 
           'UniversalPerturbation',
           'UAPDataset']



def validate_audio_attack():
    return "Audio attack module loaded successfully. Ready for adversarial attacks on ASR models."