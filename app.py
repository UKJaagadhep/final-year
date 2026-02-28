import streamlit as st
import tempfile
from pathlib import Path
from preprocessing import prepare_inference_input
from real import tamil_audio
from utils.gen import tamil_tts
from utils import getAudio
from inf import tts_inf

# --- Mapping Dictionaries ---
gender_map = {'Female': 0, 'Male': 1}
age_map = {'18-30': 0, '30-45': 1, '45-60': 2, '60+': 3}
district_map = {
    'Ariyalur': 0, 'Coimbatore': 1, 'Cuddalore': 2, 'Dharmapuri': 3, 'Erode': 4,
    'Kallakurichi': 5, 'Krishnagiri': 6, 'Mayiladuthurai': 7, 'Nagapattinam': 8,
    'Namakkal': 9, 'Perambalur': 10, 'Pudukkottai': 11, 'Salem': 12, 'Sivaganga': 13,
    'Thanjavur': 14, 'Tiruchirappalli': 15, 'Tiruppur': 16, 'Tiruvarur': 17, 'Viluppuram': 18
}

st.set_page_config(
    page_title="SANA: Tamil TTS",
    page_icon="🎤",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
h1 {
    color: #4B0082;
    text-align: center;
}
.stButton>button {
    background-color: #4B0082;
    color: white;
    font-weight: bold;
}
.stTextArea>div>textarea {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🎤 SANA: Smart Tamil TTS Assistant")

st.markdown("""
Welcome to **SANA**, the Smart Assistant for Neural Adaptive Text-to-Speech conversion in Tamil.
Select the speaker profile and type your Tamil text to generate speech.
""")

# --- Input Section ---
st.subheader("Enter Tamil Text")
tamil_text = st.text_area("Type your Tamil text here", height=150)

st.subheader("Select Speaker Profile")

#col1, col2, col3 = st.columns(3)

#with col1:
#   age = st.selectbox("Age", options=list(age_map.keys()))
#with col2:
#   gender = st.selectbox("Gender", options=list(gender_map.keys()))
#with col3:
#   accent = st.selectbox("District/Accent", options=list(district_map.keys()))


col = st.columns(1)
with col[0]:
    gender = st.selectbox("Gender", options=list(gender_map.keys()))

# --- Generate Button ---
if st.button("Generate Audio"):
    if not tamil_text.strip():
        st.error("Please enter Tamil text to generate audio.")
    else:
        st.info("Generating audio, please wait... 🎧")

        # --- Placeholder for SANA TTS Call ---
        # Replace this with your actual SANA TTS function
        def generate_tts(text, gender_id):
            # Example: create a dummy audio file
            import numpy as np
            import soundfile as sf

            metadata = {"gender":gender_id}
            text_tensor, text_mask, meta_tensor = prepare_inference_input(text, metadata)
            audio = tamil_tts(text, gender)
            #tempfile = tamil_audio(audio)
            return audio

        # Map dropdown values to IDs
        gender_id = gender_map[gender]

        # Generate audio
        audio_file = generate_tts(tamil_text, gender_id)

        st.success("✅ Audio Generated!")
        st.audio(audio_file, format="audio/wav")
        st.markdown(f"**Speaker Profile:**  Gender: {gender}")