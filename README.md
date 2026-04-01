# 🎬 VideoTranslate AI - Multilingual Video Translation Platform

A sophisticated AI-powered web application that translates video content across multiple languages using advanced machine learning models. The platform leverages Whisper for speech recognition, NLLB-200 for translation, and RAG (Retrieval-Augmented Generation) for context-aware processing.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)

---

## ✨ Features

- 🎥 **Video Upload & Processing** - Support for multiple video formats (MP4, AVI, MOV, MKV, WAV)
- 🌐 **Multilingual Translation** - Translate videos to and from numerous languages
- 🧠 **AI-Powered Intelligence** - Summarize and analyze video transcripts with Google Gemini
- 📝 **Automatic Subtitle Generation** - Generate and embed translated subtitles directly into videos
- ⚡ **Fast Processing** - Efficient algorithms for rapid video translation
- 💬 **Content Generation** - Instantly get summaries of the videos using Generative AI
- 💾 **Easy Download** - Get your translated videos with embedded subtitles

---
## 📷 Screen Shots

<img width="1443" height="828" alt="image" src="https://github.com/user-attachments/assets/7ea76da6-5c0e-429f-9ff3-19d3630e227a" />


<img width="1470" height="835" alt="image" src="https://github.com/user-attachments/assets/76f72243-b7e6-4d0f-bf6c-0275b383b10b" />


<img width="1449" height="740" alt="image" src="https://github.com/user-attachments/assets/879f6e86-7f59-4928-b93f-c7246818e186" />




## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | Flask (Python) |
| **Speech Recognition** | OpenAI Whisper |
| **Translation Model** | Facebook NLLB-200 |
| **Content Summary** | Google Gemini API (gemini-2.5-flash) |
| **Video Processing** | FFmpeg |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 |
| **ML Libraries** | PyTorch, Transformers, HuggingFace |

---

## 🏗️ Architecture

```
┌─────────────┐
│   Frontend  │  User uploads video
│  (Browser)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Flask API  │  Receives video file
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Whisper   │  Extracts audio & transcribes
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Gemini API│  Context analysis & summarization
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   NLLB-200  │  Translates content
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   FFmpeg    │  Embeds subtitles & generates video
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   User      │  Downloads translated video & reads summary
└─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11** (Highly Recommended)
  - *Note: Higher versions like 3.14+ may have compatibility issues with some ML libraries.*
- **FFmpeg** installed on your system
- **8GB+ RAM** (16GB recommended for optimal performance)
- **5GB+ free disk space** for ML models
- **Internet connection** (for initial model download)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows:**
```batch
run.bat
```

**macOS / Linux:**
```bash
chmod +x run.sh
./run.sh
```

#### Option 2: Manual Setup

1. **Clone & Enter Directory**
   ```bash
   git clone https://github.com/raghulpranxsh/CrossLingualAI.git
   cd CrossLingualAI
   ```

2. **Create Environment (Python 3.11)**
   ```bash
   # Windows
   py -3.11 -m venv venv_311
   venv_311\Scripts\activate

   # macOS / Linux
   python3.11 -m venv venv_311
   source venv_311/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run Application**
   ```bash
   python app.py
   ```

### First Run

On the first run, the application will automatically download required ML models:
- **Whisper Base Model** (~150MB)
- **NLLB-200 Translation Model** (~1.2GB)

**Note:** Initial download may take 10-15 minutes. Models are cached locally in `~/.cache/huggingface/` for subsequent runs.

---

## 📖 Usage

1. **Start the Server**
   ```bash
   python app.py
   ```

2. **Access the Application**
   - Open your browser and navigate to: `http://localhost:5001`

3. **Upload Video**
   - Click "Choose File" or drag and drop your video
   - Supported formats: MP4, AVI, MOV, MKV, WAV
   - Maximum file size: 500MB

4. **Configure Translation**
   - Select source language (or use Auto-detect)
   - Select target language
   - Click "Process Translation"

5. **Download Result**
   - Wait for processing to complete
   - Download your translated video with embedded subtitles

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web interface |
| `GET` | `/api/health` | Health check endpoint |
| `POST` | `/api/upload` | Upload and process video |
| `GET` | `/api/download/<filename>` | Download processed video |

### Example API Usage

```bash
# Health check
curl http://localhost:5001/api/health

# Upload video (using curl)
curl -X POST -F "file=@video.mp4" \
     -F "sourceLanguage=auto" \
     -F "targetLanguage=en" \
     http://localhost:5001/api/upload
```

---

## 🧠 How It Works

1. **Audio Extraction**: Video file is processed to extract audio track
2. **Speech Recognition**: Whisper transcribes audio to text in the original language
3. **Language Detection**: Automatic detection of source language (if not specified)
4. **Summary Generation**: Transcript is sent to Google Gemini to get a concise summary
5. **Translation**: NLLB-200 translates the transcribed text to target language
6. **Subtitle Generation**: SRT file is created with translated subtitles and timestamps
7. **Video Processing**: FFmpeg embeds subtitles into the original video
8. **Delivery**: User receives the translated video with embedded subtitles and a textual summary

---

## 📁 Project Structure

```
CrossLingualAI/
│
├── app.py                 # Flask backend server
├── index.html            # Frontend web interface
├── requirements.txt      # Python dependencies
├── run.sh               # Setup script (macOS/Linux)
├── run.bat              # Setup script (Windows)
├── README.md            # Project documentation
│
├── uploads/             # Temporary upload directory
└── outputs/             # Processed video output directory
```

---

## ⚙️ Configuration

### Port Configuration

By default, the server runs on port `5001`. To change this, modify `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

### Model Configuration

Models are automatically downloaded on first run. To use different Whisper models:

```python
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
```

---

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill process on port 5001
lsof -ti :5001 | xargs kill -9
```

**FFmpeg Not Found**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Out of Memory**
- Close other applications
- Use smaller Whisper model (tiny/base instead of large)
- Process shorter videos

**Models Not Downloading**
- Check internet connection
- Verify disk space (need ~2GB free)
- Check HuggingFace access

---

## 📊 Performance

- **Processing Time**: ~2-5 minutes for a 1-minute video
- **Memory Usage**: ~3-4GB during processing
- **Supported Languages**: 20+ languages via NLLB-200
- **Video Formats**: MP4, AVI, MOV, MKV, WAV

---

## 🔒 Security Notes

- Uploaded files are temporarily stored and automatically deleted after processing
- No user data is permanently stored
- All processing happens server-side
- Maximum file size limit: 500MB

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Raghul Pranesh K V**

- 🌐 GitHub: [@raghulpranxsh](https://github.com/raghulpranxsh)
- 💼 LinkedIn: [raghulpraneshkv](https://www.linkedin.com/in/raghulpraneshkv/)

---

## Acknowledgments

- OpenAI for Whisper speech recognition model
- Facebook AI for NLLB-200 translation model
- HuggingFace for Transformers and Sentence Transformers
- Flask community for the excellent web framework

