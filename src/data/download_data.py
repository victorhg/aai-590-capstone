import os
import torchaudio
from torch.utils.data import Dataset
from typing import List

# Constants
LIBRI_URL = "test-clean" # We use the small 'test-clean' set for this project

class AudioDataset(Dataset):
    """Simple wrapper to load audio from a list of paths."""
    def __init__(self, file_list: List[str]):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.file_list[idx]

def download_librispeech_sample(output_dir: str = None) -> str:
    """
    Downloads and extracts the LibriSpeech test-clean dataset using Torchaudio.
    
    Args:
        output_dir: The root directory to save data. If None, uses default project 'data' dir.
        
    Returns:
        str: The absolute path to the downloaded dataset folder.
    """
    if output_dir is None:
        # Resolve default relative to this file: src/data/../../data
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

    print(f"Checking/Downloading LibriSpeech ({LIBRI_URL}) to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Torchaudio handles downloading and extraction automatically
    # It checks md5 checksums, so it won't re-download if files exist.
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=output_dir,
        url=LIBRI_URL,
        download=True
    )
    
    # The standard path torchaudio creates is root/LibriSpeech
    final_path = os.path.join(output_dir, "LibriSpeech", LIBRI_URL)
    return final_path

if __name__ == "__main__":
    # Allow running this script directly from terminal
    download_librispeech_sample("../../data")