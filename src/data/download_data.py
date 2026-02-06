"""
Script to download LibriSpeech and CommonVoice datasets.
Note: This requires internet access and sufficient disk space.
"""

from datasets import load_dataset
import os
import shutil

def setup_directories():
    """Create directory structure for data."""
    base_dir = "src/data"
    libris_dir = os.path.join(base_dir, "librispeech")
    commonvoice_dir = os.path.join(base_dir, "commonvoice")
    
    os.makedirs(libris_dir, exist_ok=True)
    os.makedirs(commonvoice_dir, exist_ok=True)
    
    return libris_dir, commonvoice_dir

def download_librispeech(libris_dir: str):
    """
    Download LibriSpeech test-clean subset.
    Size: ~75 utterances.
    """
    print("Downloading LibriSpeech (test-clean)...")
    
    try:
        # Using Hugging Face 'datasets' library for easy download
        dataset = load_dataset("librispeech_asr", "librispeech_test_clean", split="train", streaming=True)
        
        # We will iterate and save first few items or specific splits if available
        # For this task, we download a few samples to ensure structure works.
        # In a real scenario, we might want a specific split like 'train-100'
        
        count = 0
        for i, item in enumerate(dataset):
            if count >= 75: # Limit to ensure we don't fill disk immediately
                break
            
            # Save audio
            audio_path = os.path.join(libris_dir, f"libri_{i}.flac")
            
            # Save text
            text_path = os.path.join(libris_dir, f"libri_{i}.txt")
            
            with open(text_path, "w") as f:
                f.write(item["text"].strip())
            
            # audio data is in item['audio']['array'] and path is item['audio']['path']
            # Note: item['audio']['path' points to local file if already downloaded, else None.
            # We need to save it explicitly to have our own file.
            sf.write(audio_path, item['audio']['array'], item['audio']['sampling_rate'])
            
            print(f"Downloaded {i}: {item['text'][:20]}...")
            count += 1
            
    except Exception as e:
        print(f"Error downloading LibriSpeech: {e}")

def download_commonvoice(commonvoice_dir: str):
    """
    Download a small sample from CommonVoice.
    """
    print("Downloading CommonVoice (sample)...")
    
    try:
        # Downloading a small subset of English
        dataset = load_dataset("mozilla-foundation/common_voice", "en", split="train", streaming=True)
        
        count = 0
        for i, item in enumerate(dataset):
            if count >= 20:
                break
            
            # Save audio
            audio_path = os.path.join(commonvoice_dir, f"common_{i}.flac")
            
            # Save text
            text_path = os.path.join(commonvoice_dir, f"common_{i}.txt")
            
            with open(text_path, "w") as f:
                f.write(item["sentence"].strip())
            
            sf.write(audio_path, item['audio']['array'], item['audio']['sampling_rate'])
            
            print(f"Downloaded {i}: {item['sentence'][:20]}...")
            count += 1
            
    except Exception as e:
        print(f"Error downloading CommonVoice: {e}")

def main():
    lib_dir, cv_dir = setup_directories()
    
    # Note: These downloads might take time and require significant bandwidth.
    # We'll try to download but handle interruptions gracefully.
    
    # For the purpose of this project setup, we will try to download the data.
    # If the user runs this without internet, it will fail, but the code structure is ready.
    
    download_librispeech(lib_dir)
    download_commonvoice(cv_dir)
    
    print("Data download complete.")

if __name__ == "__main__":
    main()
