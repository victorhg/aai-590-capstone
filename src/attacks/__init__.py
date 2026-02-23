# Initialize the attacks module
from .pgd import PGDAttack
from .uap import UniversalPerturbation, UAPDataset
from .cw import CWAuditoryAttack, CarliniWagnerAttack
import numpy as np

__all__ = [
    'PGDAttack',
    'UniversalPerturbation',
    'UAPDataset',
    'CWAuditoryAttack',
    'CarliniWagnerAttack',
]



def validate_audio_attack():
    return "Audio attack module loaded successfully. Ready for adversarial attacks on ASR models."



def compute_snr(original: np.ndarray, perturbed: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        original: Clean audio (numpy array).
        perturbed: Adversarial audio (numpy array).
        
    Returns:
        SNR in dB.
    """
    original = original.astype(np.float32)
    perturbed = perturbed.astype(np.float32)
    
    noise = perturbed - original
    
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)