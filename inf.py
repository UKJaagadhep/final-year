import torch
import numpy as np
import io
import soundfile as sf

def tts_inf(text_tensor, text_mask, meta_tensor) -> bytes:

    model_path = "C:\\Users\\Hp\\Documents\\final year\\best_film_model_final.pth"

    # --- Load model checkpoint (demo only) ---
    try:
        model_data = torch.load(model_path)
        print("✅ Model checkpoint loaded (demo)")
    except Exception as e:
        print("⚠️ Could not load model:", e)
        model_data = None

    sr = 16000          # sample rate
    duration_sec = 2     # 2-second dummy audio
    num_samples = int(sr * duration_sec)

    # Random waveform simulating speech
    fake_waveform = np.random.uniform(-0.3, 0.3, num_samples).astype(np.float32)

    # Apply simple envelope for fade-in/out
    t = np.linspace(0, duration_sec, num_samples)
    envelope = 0.5 * (1 - np.cos(2 * np.pi * t / duration_sec))
    fake_waveform = fake_waveform * envelope

    # Convert to WAV bytes in-memory
    buf = io.BytesIO()
    sf.write(buf, fake_waveform, sr, format="WAV")
    buf.seek(0)
    return buf.read()



