# 🚀 Quick Start Guide - SecureNews

## Method 1: Automated Script (Recommended)

### macOS / Linux:
```bash
./run.sh
```

### Windows:
```batch
run.bat
```

The script will:
- ✅ Check for Python and FFmpeg
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Start the server

---

## Method 2: Manual Setup

### Step 1: Install Prerequisites

- **Python 3.11** (Highly Recommended)
- **FFmpeg** installed (brew install ffmpeg / sudo apt install ffmpeg)

### Step 2: Create Virtual Environment
```bash
# Windows
py -3.11 -m venv venv_311
venv_311\Scripts\activate

# macOS / Linux
python3.11 -m venv venv_311
source venv_311/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

---

## 🌐 Access the Application

Once the server starts (after models finish loading), open your browser:

**URL:** http://localhost:5001

---

## ⏱️ First Run Timeline

- **Setup:** 2-5 minutes (installing packages)
- **Model Download:** 10-15 minutes (first run only)
  - Whisper model (~150MB)
  - NLLB-200 model (~1.2GB)
- **Subsequent Runs:** 1-2 minutes (models cached)

---

## ✅ Check Server Status

```bash
curl http://localhost:5001/api/health
```

Expected response when ready:
```json
{"status": "healthy", "models_loaded": true}
```

---

## 🎬 Using the Application

1. Open http://localhost:5001 in your browser
2. Upload a video file (MP4, AVI, MOV, MKV, or WAV)
3. Select source and target languages
4. Click "Translate Video"
5. Wait for processing (may take several minutes)
6. Download your translated video with embedded subtitles

---

## 🛑 Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

---

## ❓ Troubleshooting

**Port 5001 already in use?**
- The app will show an error. Kill the process using port 5001 or change the port in `app.py`

**Models not downloading?**
- Check your internet connection
- Models download to `~/.cache/huggingface/`

**Out of memory?**
- Close other applications
- The models require ~2-3GB RAM

**FFmpeg errors?**
- Ensure FFmpeg is installed: `ffmpeg -version`
- On macOS: `brew install ffmpeg`

