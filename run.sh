#!/bin/bash

# SecureNews - Run Script
# This script sets up and runs the SecureNews application

echo "🚀 Starting SecureNews Application..."
echo ""

# Check for Python 3.11 preferred
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "✅ Python 3.11 detected."
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Python 3 detected (Python 3.11 recommended)."
else
    echo "❌ Python is not installed. Please install Python 3.11 (Recommended)."
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg is not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Please install Homebrew first: https://brew.sh"
            echo "   Then run: brew install ffmpeg"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "❌ Please install FFmpeg manually for your OS"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv_311" ]; then
    echo "📦 Creating virtual environment using $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv_311
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_311/bin/activate

# Install/upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads outputs

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting Flask server..."
echo "   Open your browser and navigate to: http://localhost:5001"
echo ""
echo "⚠️  Note: First run will download ML models (Whisper, NLLB, etc.)"
echo "   This may take 10-15 minutes depending on your internet connection."
echo ""

# Run the application
python app.py

