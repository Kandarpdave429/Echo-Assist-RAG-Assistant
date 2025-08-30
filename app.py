import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from faster_whisper import WhisperModel

from rag.AIVoiceAssistant import AIVoiceAssistant
import voice_service as vs

# for ragas evaluation
import json

eval_data = []
MAX_EVAL_Q = 8


app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize AI assistant
ai_assistant = AIVoiceAssistant()

# Initialize Faster Whisper model
WHISPER_MODEL_SIZE = "small.en"
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", num_workers=2)

# Silence threshold
SILENCE_THRESHOLD = 2000


def is_silence(data, max_amplitude_threshold=SILENCE_THRESHOLD):
    return np.max(np.abs(data)) <= max_amplitude_threshold


def transcribe_audio(file_path):
    segments, _ = whisper_model.transcribe(file_path, beam_size=7)
    return ' '.join(segment.text for segment in segments)

def log_ragas_entry(question, answer, contexts):
    eval_data.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ""  # You will fill manually later
    })

    if len(eval_data) >= MAX_EVAL_Q:
        with open("ragas_eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=4, ensure_ascii=False)
        print("✅ ragas_eval.json file saved with", len(eval_data), "entries.")


# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat_text():
    """Text-based chat."""
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is empty"}), 400

    result= ai_assistant.interact_with_llm(query)
    # Log for RAGAS
    log_ragas_entry(query, result['answer'], result['retrieved_chunks'])
    return jsonify({
        "query": query,
        "answer": result['answer'],
        "retrieved_chunks": result['retrieved_chunks']
    })


@app.route("/voice", methods=["POST"])
def chat_voice():
    """Voice-based chat (WAV file upload)."""
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files['file']
    temp_path = "temp_input.wav"
    file.save(temp_path)

    # Transcribe audio
    try:
        transcription = transcribe_audio(temp_path)
        os.remove(temp_path)
    except Exception as e:
        return jsonify({"error": f"Failed to transcribe audio: {e}"}), 500

    if not transcription.strip():
        return jsonify({"error": "Audio contains no speech"}), 400

    # Get AI assistant response
   
    result = ai_assistant.interact_with_llm(transcription)
    answer = result['answer']
    retrieved_chunks = result['retrieved_chunks']
    log_ragas_entry(transcription, answer, retrieved_chunks)


    # Generate TTS audio file in /static/audio/
    os.makedirs("static/audio", exist_ok=True)
    tts_filename = f"{hash(answer)}.mp3"
    tts_path = os.path.join("static/audio", tts_filename)
    vs.play_text_to_file(answer, tts_path)  # see voice_service update below

    return jsonify({
        "query": transcription,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "tts_audio_path": f"/static/audio/{tts_filename}"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
