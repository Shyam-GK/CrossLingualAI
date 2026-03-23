import os
import tempfile
import subprocess
import whisper
import json
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from werkzeug.utils import secure_filename
import time
import google.generativeai as genai

import google.generativeai as genai

genai.configure(api_key="AIzaSyBH0LK3xqcnK2HoHa4XoD6yZLg5lPunljg")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wav'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl"
}

whisper_model = None
translation_tokenizer = None
translation_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load all ML models on startup"""
    global whisper_model, translation_tokenizer, translation_model, summarization_model
    
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    
    print("Loading translation model...")
    model_name = "facebook/nllb-200-distilled-600M"
    translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    output_audio_path = tempfile.mktemp(suffix='.wav')
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        output_audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    result = whisper_model.transcribe(audio_path)
    return result["text"], result["language"], result.get("segments", [])

def translate_text(text, source_lang_code, target_lang_code="en"):
    """Translate text using NLLB model"""
    if source_lang_code == target_lang_code:
        return text
    
    src_nllb = WHISPER_TO_NLLB.get(source_lang_code)
    tgt_nllb = WHISPER_TO_NLLB.get(target_lang_code, "eng_Latn")
    
    if not src_nllb:
        return text  # Return original if language not supported
    
    translation_tokenizer.src_lang = src_nllb
    translation_tokenizer.tgt_lang = tgt_nllb
    forced_bos_token_id = translation_tokenizer.convert_tokens_to_ids(translation_tokenizer.tgt_lang)
    
    # Split text into sentences for better translation
    sentences = text.strip().split('. ')
    if not sentences[-1]:
        sentences = sentences[:-1]
    
    translated_parts = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        try:
            inputs = translation_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            translated_tokens = translation_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            translated_text = translation_tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            translated_parts.append(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")
            translated_parts.append(sentence)  # Fallback to original
    
    return ". ".join(translated_parts)

def generate_srt_from_segments(segments, filename):
    """Generate SRT file from Whisper segments"""
    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    return filename

def generate_srt_from_text(text, filename, duration_seconds=60):
    """Generate SRT file from plain text (fallback)"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        sentences = [text]
    
    time_per_sentence = duration_seconds / len(sentences) if sentences else 5
    
    with open(filename, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sentences):
            start_time = i * time_per_sentence
            end_time = (i + 1) * time_per_sentence
            start = format_timestamp(start_time)
            end = format_timestamp(end_time)
            f.write(f"{i+1}\n{start} --> {end}\n{sentence}\n\n")
    return filename

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def embed_subtitles(video_path, srt_path, output_path):
    """Embed subtitles into video using ffmpeg"""
    # Escape the SRT path for ffmpeg
    srt_path_escaped = srt_path.replace('\\', '\\\\').replace(':', '\\:')
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"subtitles={srt_path_escaped}",
        '-c:a', 'copy',
        '-y',  # Overwrite output
        output_path
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise Exception(f"FFmpeg failed: {result.stderr}")
    
    return output_path

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': whisper_model is not None
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        source_lang = request.form.get('sourceLanguage', 'auto')
        target_lang = request.form.get('targetLanguage', 'en')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        
        file_size = os.path.getsize(video_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(video_path)
            return jsonify({'error': 'File too large (max 500MB)'}), 400
        
        print("Extracting audio...")
        audio_path = extract_audio_from_video(video_path)
        
        print("Transcribing...")
        native_text, detected_lang, segments = transcribe_audio(audio_path)
        
        if source_lang == 'auto':
            source_lang = detected_lang
        
        print("Translating...")
        translated_text = translate_text(native_text, source_lang, target_lang)
        
        print("Summarizing video content using Gemini API...")
        summary = "Summary could not be generated."
        if translated_text.strip():
            try:
                gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"Please provide a simple, layman's summary of the following video transcript. Keep it concise but cover the main points:\n\n{translated_text}"
                response = gemini_model.generate_content(prompt)
                summary = response.text.replace('*', '')  # Remove markdown asterisks for the JS alert
            except Exception as e:
                print(f"Summarization error: {e}")
        
        print("Generating subtitles...")
        srt_filename = os.path.join(OUTPUT_FOLDER, f"{filename}_subtitles.srt")
        if segments and source_lang != target_lang:
            # Translate segments individually to preserve timestamps
            translated_segments = []
            for segment in segments:
                translated_seg_text = translate_text(segment["text"], source_lang, target_lang)
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_seg_text
                })
            generate_srt_from_segments(translated_segments, srt_filename)
        elif segments:
            # No translation needed, use original segments
            generate_srt_from_segments(segments, srt_filename)
        else:
            # Fallback: estimate duration from video
            duration = 60  # Default, could extract from video metadata
            generate_srt_from_text(translated_text, srt_filename, duration)
        
        print("Embedding subtitles...")
        output_filename = f"{filename.rsplit('.', 1)[0]}_translated.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        embed_subtitles(video_path, srt_filename, output_path)
        
        os.remove(audio_path)
        os.remove(video_path)
        
        return jsonify({
            'success': True,
            'output_file': output_filename,
            'transcription': native_text,
            'translation': translated_text,
            'detected_language': detected_lang,
            'summary': summary,
            'download_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed video file"""
    file_path = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("Starting SecureNews server...")
    print("Loading models (this may take a few minutes)...")
    load_models()
    print("Models loaded! Starting server...")
    app.run(debug=True, host='0.0.0.0', port=5001)

