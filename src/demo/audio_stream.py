import numpy as np
import sounddevice as sd
from collections import deque
import queue
import threading

class AudioStream:
    """
    Handles real-time audio capture from the microphone.
    Normalizes audio to [-1, 1] and stores it in a circular buffer 
    to support continuous processing (e.g., Whisper 30s chunks).
    """
    def __init__(self, samplerate=16000, target_duration=30):
        """
        Initialize the audio stream.
        
        Args:
            samplerate (int): Audio sampling rate (Whisper requires 16000).
            target_duration (int): Duration of audio buffer in seconds.
        """
        self.samplerate = samplerate
        self.target_duration = target_duration
        self.max_samples = samplerate * target_duration
        self.buffer = deque(maxlen=self.max_samples)
        self.is_running = False
        self.stream = None
        
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for the sounddevice InputStream.
        
        Args:
            indata (numpy.ndarray): Raw audio data from microphone.
            frames (int): Number of frames requested.
            time (object): Time info object.
            status (object): Status flags (e.g., overflow).
        """
        if status:
            print(f"Audio Stream Status: {status}")
            
        # Sounddevice returns int16 by default. Convert to float32 and normalize.
        # Whisper expects float32 in range [-1.0, 1.0].
        # We assume mono input (channels=1), so we take the first column.
        audio_chunk = indata[:, 0].astype(np.float32)
        
        # Normalization: Divide by the max absolute value to keep in [-1, 1]
        # If audio is silent, max is 0, which would cause division by zero. 
        # However, typically input is >0.0. 
        # To be safe:
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0:
            audio_chunk /= max_val
            
        self.buffer.extend(audio_chunk)

    def start(self):
        """Start the microphone stream."""
        if self.is_running:
            print("Stream is already running.")
            return
            
        print(f"Starting Audio Stream (16kHz)... Please allow microphone permission.")
        self.is_running = True
        blocksize = int(self.samplerate * 0.5) # Block size ~0.5 seconds
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate, 
                channels=1, 
                blocksize=blocksize, 
                callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting stream: {e}")
            self.is_running = False

    def stop(self):
        """Stop the microphone stream."""
        if not self.is_running:
            return
            
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_running = False
        print("Audio Stream stopped.")

    def get_audio(self, length=None):
        """
        Retrieve audio data from the buffer.
        
        Args:
            length (int): Number of samples to return. If None, return all available.
            
        Returns:
            numpy.ndarray: Normalized audio data (float32, [-1, 1]).
        """
        if not self.buffer:
            return np.array([])
            
        data = np.array(self.buffer)
        
        if length is not None:
            # If request is longer than available, pad with zeros (optional, 
            # but good for inference stability)
            if len(data) < length:
                padding = np.zeros(length - len(data))
                data = np.concatenate([data, padding])
            else:
                data = data[:length]
                
        return data

    def get_duration(self):
        """Get the current duration of audio in the buffer in seconds."""
        return len(self.buffer) / self.samplerate
