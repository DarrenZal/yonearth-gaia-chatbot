# Transcription Setup Guide

## ✅ STATUS: COMPLETE (October 2025)

**All 172 YonEarth podcast episodes now have word-level timestamps!**

- ✅ 172/172 episodes transcribed (100% coverage)
- ✅ 14 episodes from YouTube (broken/missing audio)
- ✅ 158 episodes from original audio
- ✅ Only episode #26 missing (doesn't exist in series)
- ✅ Ready for 3D map navigation with precise timestamps

**Transcripts location**: `/data/transcripts/episode_*.json`

---

This guide explains the setup used to complete the transcription with precise timestamps.

## System Requirements

### 1. Install FFmpeg (System Package)

FFmpeg is required for audio processing:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### 2. Install Python Dependencies

```bash
pip install -r requirements-transcription.txt
```

This installs:
- **Whisper** (OpenAI) - Accurate transcription with word-level timestamps
- **PyAnnote Audio** - Speaker diarization (who spoke when)
- **PyTorch** - Deep learning framework
- Audio processing libraries

## Configuration

### 1. Get HuggingFace Token

1. Create account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Create a new token (read access is enough)

### 2. Accept Model Licenses

You need to accept licenses for two models:

1. **Main model**: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"

2. **Dependency model**: https://huggingface.co/pyannote/speaker-diarization-community-1
   - Click "Agree and access repository"

### 3. Set Environment Variable

Add to your `.env` file:

```bash
HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Test on One Episode

Test the system on episode 1 before processing all episodes:

```bash
python3 scripts/retranscribe_episodes_with_timestamps.py test
```

This will:
- Download episode 1 audio
- Transcribe with Whisper (word-level timestamps)
- Identify speakers with diarization
- Save results with segments containing:
  - `start`: Exact timestamp (seconds)
  - `end`: Exact timestamp (seconds)
  - `text`: Transcript text
  - `speaker`: SPEAKER_00, SPEAKER_01, etc.
  - `words`: Word-level timestamps

### Process All Episodes

After successful test:

```bash
# Full run (will take several hours for 172 episodes)
python3 scripts/retranscribe_episodes_with_timestamps.py

# Or process in batches:
python3 scripts/retranscribe_episodes_with_timestamps.py range 1 50
python3 scripts/retranscribe_episodes_with_timestamps.py range 51 100
python3 scripts/retranscribe_episodes_with_timestamps.py range 101 172
```

### Configuration Options

Set in `.env` file:

```bash
# Whisper model size (base, small, medium, large)
# Larger = more accurate but slower
WHISPER_MODEL=base

# Keep audio files after transcription (for debugging)
KEEP_AUDIO_FILES=false
```

## Output Format

The script updates transcript JSON files with:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 5.2,
      "text": "Welcome to the YonEarth podcast.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Welcome", "start": 0.5, "end": 1.0},
        {"word": "to", "start": 1.0, "end": 1.1}
      ]
    }
  ],
  "audio_transcription_metadata": {
    "whisper_model": "base",
    "language": "en",
    "duration": 3241.5,
    "speakers_detected": 2,
    "segments_count": 485,
    "diarization_available": true
  }
}
```

## Performance Notes

- **Speed**: ~2-5 minutes per episode (depending on CPU/GPU)
- **Accuracy**: Whisper base model is ~95% accurate
- **Storage**: Audio files temporarily use ~40MB each
- **Memory**: ~4GB RAM per episode during processing

## GPU Acceleration (Optional)

If you have NVIDIA GPU with CUDA:

```bash
# Verify CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"

# If True, transcription will automatically use GPU
# This is 5-10x faster than CPU
```

## Troubleshooting

### FFmpeg Not Found

```bash
# Install FFmpeg
sudo apt-get install -y ffmpeg
```

### HuggingFace 403 Error

Make sure you've accepted both model licenses:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/speaker-diarization-community-1

### Out of Memory

If you get memory errors:
1. Close other applications
2. Process in smaller batches
3. Use smaller Whisper model: `WHISPER_MODEL=base`

### Audio Download Fails

Some episodes may have broken audio URLs. The script will log errors and continue with other episodes.
