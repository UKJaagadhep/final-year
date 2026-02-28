from transformers import AutoModel
import soundfile as sf

# Load IndicF5 model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Tamil reference
ref_audio_path = "C:\\Users\\Hp\\Downloads\\download.wav"
ref_text = "அப்படி விடும் போது நம்ம நிலத்தொட்டி நெக்ஸ்ட் சின்ன பாத்திரம் முதல் கொண்டு நம்ம தண்ணியை வந்து சேமித்து வச்சுட்டோம் அப்படின்னா அந்த தண்ணி வந்து அந்த அந்த விசேஷத்துக்கு வந்து ரொம்ப ரொம்ப யூஸ்ஃபுல்லாக இருக்கும்"
input_text = "இன்று நல்ல வானிலை இருக்கிறது."

# Generate Tamil speech in cloned voice
audio = model(
    input_text,
    ref_audio_path=ref_audio_path,
    ref_text=ref_text,
    language="ta",   # Tamil
    split_sentence=False
)

# Save output
sf.write("generated_tamil.wav", audio["audio"], 16000)
print("✅ Generated Tamil voice cloned audio saved as generated_tamil.wav")
