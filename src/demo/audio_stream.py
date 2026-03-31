"""
Real-time Audio Stream Module for Whisper ASR
Handles microphone input, buffering, and normalization.
"""
import sounddevice as sd
import queue
import threading
import numpy as np
import time
from typing import Optional

class AudioStream:
    """
    Manages real-time audio input for the Whisper model.
    Configured for Whisper's native 16kHz sampling rate.
    """
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1, 
                 chunk_size: int = 4096, 
                 buffer_duration: float = 30.0):
        """
        Args:
            sample_rate (int): Audio sampling rate (Whisper default: 16000).
            channels (int): Number of audio channels (1 for mono).
            chunk_size (int): Number of frames per block (e.g., 4096).
            buffer_duration (float): Duration of the internal buffer in seconds.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.target_samples = int(sample_rate * buffer_duration)
        
        # Buffer to store raw audio samples
        self.buffer = np.zeros(self.target_samples, dtype=np.float32)
        
        # Queue for non-blocking audio retrieval
        self.audio_queue = queue.Queue()
        
        # Thread for reading
        self.is_recording = False
        self.stream = None
        self.thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for the PyAudio stream.
        Processes data, normalizes it, and updates the buffer.
        """
        if status:
            print(f"Audio Stream Status: {status}")
            
        # Convert to float32 numpy array
        audio_chunk = indata[:, 0].astype(np.float32)
        
        # Normalize to [-1, 1] if necessary (sounddevice usually returns normalized int16)
        if audio_chunk.dtype != np.float32:
             # Simple normalization for common input types
             audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        
        # Update circular buffer
        # Calculate how much we need to shift
        chunk_samples = len(audio_chunk)
        
        # Shift existing buffer to the left
        self.buffer = np.roll(self.buffer, -chunk_samples)
        
        # Add new chunk to the end
        self.buffer[-chunk_samples:] = audio_chunk
        
        # Store in queue for the main thread to process if needed
        self.audio_queue.put(audio_chunk)

    def start(self):
        """Starts the audio stream."""
        if self.is_recording:
            return

        self.is_recording = True
        
        # Define stream callback
        # Note: We use sounddevice which is a wrapper over PortAudio
        try:
            # sounddevice.InputStream uses a callback approach
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
                blocksize=self.chunk_size
            )
            self.stream.start()
            print(f"Audio stream started at {self.sample_rate}Hz.")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_recording = False

    def stop(self):
        """Stops the audio stream."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Audio stream stopped.")

    def get_audio(self) -> Optional[np.ndarray]:
        """
        Retrieves the current buffer of audio.
        Returns a numpy array of shape (num_samples,) or None if stopped.
        """
        if not self.is_recording:
            return None
            
        # Ensure buffer is full (wait a moment for startup)
        if len(self.buffer) < self.target_samples * 0.5:
             return None
             
        return self.buffer.copy()

    def get_audio_queue(self) -> Optional[np.ndarray]:
        """
        Helper to get the latest chunk from the queue if needed for real-time display.
        """
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Ensures audio is normalized to [-1.0, 1.0] float32.
        """
        # Clip to avoid overflow
        audio = np.clip(audio, -1.0, 1.0)
        # Convert to float32 if not already
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

if __name__ == "__main__":
    # Simple test script to verify microphone access
    print("Testing Audio Stream...")
    print("Please speak into the microphone now...")
    
    stream = AudioStream(buffer_duration=3.0)
    stream.start()
    
    try:
        while stream.is_recording:
            data = stream.get_audio()
            if data is not None:
                # Calculate RMS for volume visualization
                rms = np.sqrt(np.mean(data**2))
                print(f"Buffer Status: {len(data)} samples | RMS: {rms:.4f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        stream.stop()
