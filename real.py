import numpy as np
import librosa
import soundfile as sf
import io
from scipy.signal import butter, lfilter

def tamil_audio(audio_bytes: bytes) -> bytes:
    def lowpass_filter(data, cutoff, sr, order=4):
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return lfilter(b, a, data)

    # Load from in-memory MP3 bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Downsample a bit
    target_sr = 16000
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Apply soft low-pass filter (slightly muffled)
    y = lowpass_filter(y, cutoff=5500, sr=target_sr)

    # Add small pitch modulation (natural imperfection)
    t = np.linspace(0, len(y) / target_sr, len(y))
    mod = 1 + 0.005 * np.sin(2 * np.pi * 4 * t)
    indices = np.minimum(np.arange(len(y)) * mod, len(y) - 1)
    y = np.interp(np.arange(len(y)), indices, y)

    # Add very light noise
    noise = np.random.normal(0, 0.002, y.shape)
    y = y + noise

    # Light compression and normalization
    rms = np.sqrt(np.mean(y**2))
    y = y / (rms * 4)
    y = y / np.max(np.abs(y))

    # Write degraded audio to memory
    buf = io.BytesIO()
    sf.write(buf, y, target_sr, format="WAV")
    buf.seek(0)
    return buf.read()