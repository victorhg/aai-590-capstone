"""
Audio Stream Module for Real-time Transcription
Handles microphone input, circular buffering for 30s audio chunks,
and normalization required by Whisper.
"""

import queue
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import time

class AudioStream:
    """
    A class to handle real-time audio capture and buffering.
    Ensures 16kHz sampling rate and normalization.
    """
    def __init__(self, channels=1, rate=16000, buffer_duration=30.0):
        """
        Initialize audio stream.
        
        Args:
            channels (int): Number of audio channels (1 for mono).
            rate (int): Sampling rate in Hz (Whisper requirement: 16000).
            buffer_duration (float): Duration of the circular buffer in seconds.
        """
        self.channels = channels
        self.rate = rate
        self.buffer_duration = buffer_duration
        self.sample_len = int(rate * buffer_duration)
        
        # Thread-safe queue for audio chunks
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Circular buffer storage (list of numpy arrays)
        self.buffer = []
        self.buffer_size = 0
        
        self.is_running = False
        self.stream = None
        
        # Lock for thread-safe buffer operations
        self.buffer_lock = threading.Lock()

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for sounddevice stream.
        Normalizes input and adds to the circular buffer.
        """
        if status:
            print(f"Audio Stream Error: {status}")
        
        # Normalize audio to [-1.0, 1.0]
        # Reshape for mono if stereo is captured
        if indata.shape[1] > 1:
            indata = indata[:, 0]
            
        # Basic normalization (RMS scaling)
        norm_factor = np.sqrt(np.mean(indata**2))
        if norm_factor > 0:
            indata = indata / norm_factor
            
        # Clip to prevent overflow
        indata = np.clip(indata, -1.0, 1.0)
        
        with self.buffer_lock:
            self.buffer.append(indata)
            self.buffer_size += len(indata)
            
            # Remove audio that has been dequeued from the main processing loop
            # (We estimate the amount dequeued based on buffer duration)
            # If we have more than needed, trim from the front
            while self.buffer_size > self.sample_len:
                # Remove oldest chunk (first in list)
                oldest = self.buffer.pop(0)
                self.buffer_size -= len(oldest)

    def start(self):
        """Start the microphone stream."""
        if self.is_running:
            return

        print(f"Starting Audio Stream @ {self.rate}Hz...")
        
        # Use a large block size to reduce callback overhead
        blocksize = int(self.rate * 0.1) 

        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.rate,
            callback=self.audio_callback,
            blocksize=blocksize
        )
        
        self.stream.start()
        self.is_running = True
        
        # Start a background thread to handle buffer trimming if the stream is fast
        # (The logic is primarily in the callback, but we ensure the queue is managed)
        self._buffer_processor = threading.Thread(target=self._process_buffer, daemon=True)
        self._buffer_processor.start()

    def _process_buffer(self):
        """
        Background thread to handle buffer management.
        If the buffer grows too large due to lag, trim it.
        """
        while self.is_running:
            time.sleep(0.1)
            with self.buffer_lock:
                # If buffer is significantly larger than needed, trim front
                # Threshold: buffer_duration + 0.5 seconds
                if self.buffer_size > self.sample_len + int(self.rate * 0.5):
                    removed = self.buffer.pop(0)
                    self.buffer_size -= len(removed)

    def get_audio_chunk(self):
        """
        Retrieve a 30-second audio chunk from the circular buffer.
        Returns a single numpy array or None if buffer is not full.
        """
        with self.buffer_lock:
            if self.buffer_size < self.sample_len:
                return None
            
            # Concatenate all chunks in the buffer
            audio_data = np.concatenate(self.buffer, axis=0)
            
            # If audio is slightly shorter (e.g., just started), pad with zeros or return as is
            # Ideally, we wait until it fills up for a clean segment
            
            # Reset buffer
            self.buffer = []
            self.buffer_size = 0
            
            return audio_data

    def stop(self):
        """Stop the stream and release resources."""
        print("Stopping Audio Stream...")
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stream = None

def save_audio_test(audio_data, filename="test_input.wav"):
    """Helper to save raw audio for testing."""
    if audio_data is not None and len(audio_data) > 0:
        sf.write(filename, audio_data, 16000)
        print(f"Saved test audio to {filename}")

if __name__ == "__main__":
    # Test the stream
    print("Testing Audio Stream Module...")
    stream = AudioStream(buffer_duration=5.0) # Test with 5s for faster demo
    stream.start()
    
    print("Listening... (Press Ctrl+C to stop)")
    try:
        while True:
            chunk = stream.get_audio_chunk()
            if chunk is not None:
                save_audio_test(chunk, "test_chunk.wav")
                print(f"Captured {len(chunk)} samples")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
