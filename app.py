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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the API key from the environment
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment variables!")
genai.configure(api_key=api_key)

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

# edge-tts voice mapping per language
EDGE_TTS_VOICES = {
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ko": "ko-KR-SunHiNeural",
    "ar": "ar-EG-SalmaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "it": "it-IT-ElsaNeural",
    "nl": "nl-NL-ColetteNeural",
}

whisper_model = None
translation_tokenizer = None
translation_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load all ML models on startup"""
    global whisper_model, translation_tokenizer, translation_model
    
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
    """Embed subtitles into video using ffmpeg without re-encoding"""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', srt_path,
        '-c', 'copy',
        '-c:s', 'mov_text',
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

# ──────────────────────────────────────────────
#  VIDEO DUBBING HELPERS
# ──────────────────────────────────────────────

def download_youtube_video(url):
    """Download a YouTube video using yt-dlp and return local path"""
    import yt_dlp
    output_template = os.path.join(UPLOAD_FOLDER, '%(id)s.%(ext)s')
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get('id', 'video')
        file_path = os.path.join(UPLOAD_FOLDER, f"{video_id}.mp4")
        # yt-dlp may produce a differently-named file; find it
        if not os.path.exists(file_path):
            for fname in os.listdir(UPLOAD_FOLDER):
                if video_id in fname:
                    file_path = os.path.join(UPLOAD_FOLDER, fname)
                    break
    return file_path

async def _generate_tts_async(text, voice, output_path):
    """Internal async edge-tts call"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_tts_for_text(text, target_lang, output_path):
    """Generate TTS audio for given text using edge-tts"""
    import asyncio
    voice = EDGE_TTS_VOICES.get(target_lang, "en-US-JennyNeural")
    if not text.strip():
        _generate_silence_wav(output_path, 0.5)
        return output_path
    try:
        asyncio.run(_generate_tts_async(text, voice, output_path))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate_tts_async(text, voice, output_path))
        loop.close()
    return output_path

def _generate_silence_wav(path, duration_sec):
    """Create a short silent WAV file"""
    cmd = [
        'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=r=24000:cl=mono',
        '-t', str(duration_sec), '-y', path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def get_audio_duration(path):
    """Return duration of audio file in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'json', path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except Exception:
        return 1.0

def speed_adjust_audio(input_path, output_path, target_duration):
    """Speed-adjust TTS audio to fit within target_duration using ffmpeg atempo"""
    actual_duration = get_audio_duration(input_path)
    if actual_duration <= 0 or target_duration <= 0:
        import shutil
        shutil.copy(input_path, output_path)
        return output_path
    
    ratio = actual_duration / target_duration
    # atempo supports 0.5–2.0; clamp to valid range
    ratio = max(0.5, min(2.0, ratio))
    
    cmd = [
        'ffmpeg', '-i', input_path,
        '-filter:a', f'atempo={ratio:.4f}',
        '-y', output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        import shutil
        shutil.copy(input_path, output_path)
    return output_path

def build_dubbed_audio(translated_segments, target_lang, video_duration):
    """
    For each segment, generate TTS, speed-match to segment duration,
    then overlay onto a silent timeline matching video duration.
    """
    from pydub import AudioSegment

    timeline = AudioSegment.silent(duration=int(video_duration * 1000))  # ms

    for i, seg in enumerate(translated_segments):
        text = seg['text'].strip()
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        seg_duration = (end_ms - start_ms) / 1000.0

        if not text:
            continue

        tts_raw_path = tempfile.mktemp(suffix=f'_tts_{i}.mp3')
        tts_adj_path = tempfile.mktemp(suffix=f'_tts_adj_{i}.wav')

        try:
            generate_tts_for_text(text, target_lang, tts_raw_path)
            speed_adjust_audio(tts_raw_path, tts_adj_path, seg_duration)
            seg_audio = AudioSegment.from_file(tts_adj_path)
            # Trim if TTS is slightly longer
            if len(seg_audio) > int(seg_duration * 1000) + 200:
                seg_audio = seg_audio[:int(seg_duration * 1000)]
            timeline = timeline.overlay(seg_audio, position=start_ms)
        except Exception as e:
            print(f"TTS error for segment {i}: {e}")
        finally:
            for p in [tts_raw_path, tts_adj_path]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    final_audio_path = tempfile.mktemp(suffix='_dubbed.wav')
    timeline.export(final_audio_path, format='wav')
    return final_audio_path

def replace_video_audio(video_path, new_audio_path, output_path):
    """Replace video's audio track with new_audio_path, keeping original audio at low volume using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', new_audio_path,
        '-filter_complex', '[0:a]volume=0.15[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2:duration=first[a]',
        '-map', '0:v:0',      # video from first input
        '-map', '[a]',        # mixed audio
        '-c:v', 'copy',       # keep original video stream
        '-c:a', 'aac',        # encode new audio as AAC
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"FFmpeg replace audio error: {result.stderr}")
        raise Exception(f"FFmpeg failed: {result.stderr}")
    return output_path

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except Exception:
        return 60.0

# ──────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint for video summaries"""
    try:
        data = request.json
        context = data.get('context', '')
        question = data.get('question', '')
        
        if not context or not question:
            return jsonify({'error': 'Context and question are required'}), 400
            
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Context from a video:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:"
        
        response = gemini_model.generate_content(prompt)
        return jsonify({'answer': response.text})
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': whisper_model is not None
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing (subtitle embedding)"""
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
                summary = response.text.replace('*', '')
            except Exception as e:
                print(f"Summarization error: {e}")
        
        print("Generating subtitles...")
        srt_filename = os.path.join(OUTPUT_FOLDER, f"{filename}_subtitles.srt")
        if segments and source_lang != target_lang:
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
            generate_srt_from_segments(segments, srt_filename)
        else:
            duration = 60
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


@app.route('/api/translate-audio-video', methods=['POST'])
def translate_audio_video():
    """
    VIDEO DUBBING endpoint:
    - Accepts file upload OR YouTube URL
    - Transcribes audio (Whisper) — auto-detects source language
    - Translates per-segment (NLLB)
    - Generates neural TTS audio (edge-tts) per segment, speed-matched
    - Replaces original audio with TTS track (FFmpeg)
    - Returns dubbed MP4 for download
    """
    video_path = None
    audio_path = None
    dubbed_audio_path = None
    cleanup_video = False

    try:
        target_lang = request.form.get('targetLanguage', 'en')
        youtube_url = request.form.get('youtubeUrl', '').strip()

        # ── Input: YouTube URL or file upload ──
        if youtube_url:
            print(f"Downloading YouTube video: {youtube_url}")
            video_path = download_youtube_video(youtube_url)
            cleanup_video = True
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            filename = secure_filename(file.filename)
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(video_path)
            cleanup_video = True
            if os.path.getsize(video_path) > MAX_FILE_SIZE:
                os.remove(video_path)
                return jsonify({'error': 'File too large (max 500MB)'}), 400
        else:
            return jsonify({'error': 'Provide a video file or YouTube URL'}), 400

        # ── Step 1: Extract audio ──
        print("Extracting audio...")
        audio_path = extract_audio_from_video(video_path)

        # ── Step 2: Transcribe ──
        print("Transcribing with Whisper (auto-detect language)...")
        native_text, detected_lang, segments = transcribe_audio(audio_path)
        print(f"Detected language: {detected_lang}")

        # ── Step 3: Translate each segment ──
        print(f"Translating {detected_lang} → {target_lang}...")
        if detected_lang == target_lang:
            translated_segments = [{'start': s['start'], 'end': s['end'], 'text': s['text']} for s in segments]
            translated_text = native_text
        else:
            translated_segments = []
            for seg in segments:
                t = translate_text(seg['text'], detected_lang, target_lang)
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': t
                })
            translated_text = ' '.join(s['text'] for s in translated_segments)

        # ── Step 4: Get video duration ──
        video_duration = get_video_duration(video_path)

        # ── Step 5: Build dubbed audio track ──
        print("Generating neural TTS dubbed audio track...")
        dubbed_audio_path = build_dubbed_audio(translated_segments, target_lang, video_duration)

        # ── Step 6: Replace audio in video ──
        print("Muxing new audio into video...")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_dubbed_{target_lang}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        replace_video_audio(video_path, dubbed_audio_path, output_path)

        return jsonify({
            'success': True,
            'output_file': output_filename,
            'transcription': native_text,
            'translation': translated_text,
            'detected_language': detected_lang,
            'download_url': f'/api/download/{output_filename}'
        })

    except Exception as e:
        print(f"Dubbing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [audio_path, dubbed_audio_path]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if cleanup_video and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


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
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)

