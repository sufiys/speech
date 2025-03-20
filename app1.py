import streamlit as st
import whisper
import jiwer
import os

# Set Streamlit Page Config
st.set_page_config(page_title="Student Reading Analysis", layout="wide")

st.title("📖 Student Reading Analysis App")
st.write("Upload a recording of a student reading and provide the reference text. The app will analyze the reading and generate a detailed scorecard.")

# Upload audio file
uploaded_file = st.file_uploader("🎤 Upload a reading audio file", type=["mp3", "wav"])

# Input reference text
reference_text = st.text_area("📜 Enter the reference text:")

if uploaded_file and reference_text.strip():
    # Save uploaded file temporarily
    temp_audio_path = "temp_audio.mp3"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load Whisper Model
    with st.spinner("Transcribing audio..."):
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path)
        transcription = result["text"]
    
    st.subheader("📝 Transcription")
    st.write(transcription)

    # Compute Word Error Rate (WER)
    wer = jiwer.wer(reference_text, transcription)  # ✅ FIXED WER Calculation

    # Compute other errors (insertions, deletions, substitutions)
    error_analysis = jiwer.process_words(reference_text, transcription)

    # Debugging: Print error_analysis
    st.subheader("🔍 Error Analysis Debug Info")
    st.json(error_analysis)

    # Ensure keys exist before accessing them
    omissions = len(error_analysis.get("deletions", []))
    insertions = len(error_analysis.get("insertions", []))
    substitutions = len(error_analysis.get("substitutions", []))

    # Calculate additional metrics
    total_words = len(reference_text.split())
    total_errors = omissions + insertions + substitutions
    accuracy = max(0, round(((total_words - total_errors) / total_words) * 100, 2))
    
    # Scorecard Table
    st.subheader("📊 Scorecard")
    scorecard = {
        "Errors": total_errors,
        "Omissions": omissions,
        "Insertions": insertions,
        "Substitutions": substitutions,
        "Scored Word Count": total_words,
        "Accuracy": f"{accuracy}%",
        "WER": round(wer, 4)  # ✅ Correctly computed WER
    }
    st.table(scorecard)

    # Cleanup temp file
    os.remove(temp_audio_path)
else:
    st.warning("Please upload an audio file and enter reference text to proceed.")
