import streamlit as st
import whisper
import jiwer
import os
from io import BytesIO
import soundfile as sf

# Set Streamlit Page Config
st.set_page_config(page_title="Student Reading Analysis", layout="wide")

st.title("📖 Student Reading Analysis App")
st.write("Upload or record a reading and provide the reference text. The app will analyze the reading and generate a detailed scorecard.")

# Select Input Method
option = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])

audio_data = None

# 📤 **Option 1: Upload an Audio File**
if option == "Upload Audio":
    uploaded_file = st.file_uploader("🎤 Upload a reading audio file", type=["mp3", "wav"])
    if uploaded_file:
        audio_data = uploaded_file.read()

# 🎙️ **Option 2: Record Audio Using JavaScript**
elif option == "Record Audio":
    st.write("Click the button below to record:")

    # JavaScript for recording audio
    audio_recorder_html = """
    <script>
    let mediaRecorder;
    let audioChunks = [];

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                mediaRecorder.onstop = () => {
                    let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    let reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        let base64Audio = reader.result.split(',')[1];
                        fetch('/upload_audio', { method: 'POST', body: JSON.stringify({ audio: base64Audio }) });
                    };
                };
                mediaRecorder.start();
            });
    }

    function stopRecording() {
        mediaRecorder.stop();
    }
    </script>

    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    """

    st.components.v1.html(audio_recorder_html, height=100)

# 📜 Enter Reference Text
reference_text = st.text_area("📜 Enter the reference text:")

if audio_data and reference_text.strip():
    # Save uploaded/recorded file
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data)

    # Load Whisper Model
    with st.spinner("Transcribing audio..."):
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path)
        transcription = result["text"]

    st.subheader("📝 Transcription")
    st.write(transcription)

    # Compute Word Error Rate (WER)
    wer = jiwer.wer(reference_text, transcription)

    # Compute errors
    error_analysis = jiwer.compute_measures(reference_text, transcription)

    # ✅ Accessing errors correctly
    omissions = error_analysis["deletions"]
    insertions = error_analysis["insertions"]
    substitutions = error_analysis["substitutions"]

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
        "WER": round(wer, 4)
    }
    st.table(scorecard)

    # Cleanup temp file
    os.remove(temp_audio_path)
else:
    st.warning("Please upload/record an audio file and enter reference text to proceed.")
