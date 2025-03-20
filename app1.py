import streamlit as st
import whisper
import soundfile as sf
import tempfile
from jiwer import wer, cer, process_words

# Load Whisper model
model = whisper.load_model("base")

st.title("Student Reading Analysis System")

# File uploader for audio
audio_file = st.file_uploader("Upload Student's Reading (MP3/WAV)", type=["mp3", "wav"])

# Text area for reference passage
reference_text = st.text_area("Enter the text the student is reading")

if audio_file and reference_text:
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    # Transcribe audio
    result = model.transcribe(temp_audio_path)
    transcribed_text = result["text"]

    # Compute errors
    error_analysis = process_words(reference_text, transcribed_text)
    omissions = len(error_analysis["deletions"])
    insertions = len(error_analysis["insertions"])
    substitutions = len(error_analysis["substitutions"])
    total_words = len(reference_text.split())
    
    accuracy = max(0, 100 - (wer(reference_text, transcribed_text) * 100))
    wpm = len(transcribed_text.split()) / (result["segments"][-1]["end"] / 60)  # Words per minute

    # Display results
    st.subheader("Scorecard")
    st.write(f"**Errors:** {omissions + insertions + substitutions}")
    st.write(f"**Omissions:** {omissions}")
    st.write(f"**Insertions:** {insertions}")
    st.write(f"**Mispronunciations/Substitutions:** {substitutions}")
    st.write(f"**Total Words Read:** {len(transcribed_text.split())}")
    st.write(f"**WPM (Words Per Minute):** {wpm:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}%")

    # Display transcribed text
    st.subheader("Transcribed Text")
    st.text(transcribed_text)
