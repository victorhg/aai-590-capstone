import os
import torchaudio
from torch.utils.data import Dataset
from typing import List

# Constants
LIBRI_SPEECH_URL = "http://www.openslr.org/resources/librispeech"
COMMON_VOICE_BASE_URL = "https://voice.mozilla.org/en/datasets"

class AudioDataset(Dataset):
    """Generic Dataset for audio files."""
    def __init__(self, file_list: List[str]):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.file_list[idx]

def download_librispeech_sample(output_dir: str, n_samples: int = 10):
    """
    Placeholder for LibriSpeech download.
    LibriSpeech is large (hundreds of GB). We download a small subset if possible,
    or instruct the user to manually download.
    
    Note: For this project, manual download of 'test-clean' is recommended due to size.
    """
    libri_dir = os.path.join(output_dir, "librispeech")
    os.makedirs(libri_dir, exist_ok=True)
    
    print(f"LibriSpeech data should ideally be downloaded manually to: {libri_dir}")
    print("Dataset structure: librispeech/test-clean/...\n")
    
    # Ideally, we would download specific files here using wget or aria2c.
    # Since we cannot run shell commands directly, we return the path structure.
    return libri_dir

def download_common_voice_sample(output_dir: str):
    """
    Placeholder for CommonVoice download.
    """
    print("CommonVoice download link: https://voice.mozilla.org/en/datasets")
    print("Please download the 'en' (English) subset for this project.")
    return output_dir

if __name__ == "__main__":
    print("Download utilities loaded.")
