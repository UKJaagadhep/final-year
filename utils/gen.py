from elevenlabs.client import ElevenLabs
client = ElevenLabs(api_key="b44e2845f7fc1e3d9063936c158c8d9fdb994916dfd8480e4648b4276f58d8a8")

VOICE_IDS = {
    "male": "yt40uMsmnhVftG8ngHsz",
    "female": "gCr8TeSJgJaeaIoV4RWH"
}

def tamil_tts(text: str, gender: str = "male") -> bytes:
    gender = gender.lower()

    voice_id = VOICE_IDS[gender]

    audio_gen = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    audio_bytes = b"".join(audio_gen)
    return audio_bytes


