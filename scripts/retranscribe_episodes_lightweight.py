"""
Re-transcribe YonEarth podcast episodes with PRECISE TIMESTAMPS using Whisper

This lightweight version focuses on what we need most: accurate timestamps for
the 3D map navigation. Speaker diarization is skipped to avoid memory issues.

If you want speaker labels, use the full version with ENABLE_DIARIZATION=true
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import whisper
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Configuration
TRANSCRIPTS_DIR = Path("/Users/darrenzal/projects/yonearth-gaia-chatbot/data/transcripts")
TEMP_AUDIO_DIR = Path("/tmp/yonearth_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # base, small, medium, large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EpisodeTranscriber:
    """Transcribe episodes with precise timestamps"""

    def __init__(self):
        logger.info(f"Initializing transcriber on device: {DEVICE}")
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")

        self.whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

        logger.info("✓ Transcriber initialized successfully")

    def download_audio(self, url: str, episode_number: int) -> Path:
        """Download audio file from URL"""
        audio_path = TEMP_AUDIO_DIR / f"episode_{episode_number}.mp3"

        # Skip if already downloaded
        if audio_path.exists():
            logger.info(f"Using cached audio: {audio_path}")
            return audio_path

        logger.info(f"Downloading audio from: {url}")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✓ Downloaded: {audio_path} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return audio_path

        except Exception as e:
            logger.error(f"Failed to download audio: {e}")
            raise

    def transcribe_with_whisper(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe audio with Whisper (word-level timestamps)"""
        logger.info("Transcribing with Whisper...")

        result = self.whisper_model.transcribe(
            str(audio_path),
            word_timestamps=True,  # Enable word-level timestamps
            verbose=False
        )

        logger.info(f"✓ Transcription complete ({len(result['segments'])} segments)")
        return result

    def transcribe_episode(self, episode_number: int, audio_url: str, skip_if_exists: bool = True) -> Dict[str, Any]:
        """Complete transcription pipeline for one episode"""
        logger.info(f"\n{'='*70}")
        logger.info(f"TRANSCRIBING EPISODE {episode_number}")
        logger.info(f"{'='*70}")

        output_file = TRANSCRIPTS_DIR / f"episode_{episode_number}.json"

        # Skip if already processed with segments
        if skip_if_exists and output_file.exists():
            with open(output_file) as f:
                existing_data = json.load(f)
                if 'segments' in existing_data and existing_data.get('segments'):
                    logger.info(f"✓ Episode {episode_number} already has segments, skipping")
                    return existing_data

        try:
            # Download audio
            audio_path = self.download_audio(audio_url, episode_number)

            # Transcribe with Whisper
            whisper_result = self.transcribe_with_whisper(audio_path)

            # Convert Whisper segments to our format
            segments = []
            for segment in whisper_result['segments']:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'speaker': None,  # No speaker info in lightweight version
                    'words': segment.get('words', [])  # Word-level timestamps preserved
                })

            # Load existing episode data
            if output_file.exists():
                with open(output_file) as f:
                    episode_data = json.load(f)
            else:
                episode_data = {
                    "episode_number": episode_number,
                    "audio_url": audio_url,
                }

            # Generate full transcript from segments
            full_transcript = "\n\n".join([
                f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}"
                for seg in segments
            ])

            # Update episode data
            episode_data.update({
                "segments": segments,
                "full_transcript": full_transcript,
                "audio_transcription_metadata": {
                    "whisper_model": WHISPER_MODEL,
                    "language": whisper_result.get('language', 'en'),
                    "duration": whisper_result.get('duration', 0),
                    "speakers_detected": 0,  # No diarization
                    "segments_count": len(segments),
                    "diarization_available": False,
                    "audio_url": audio_url,
                    "word_timestamps": True
                }
            })

            # Save updated data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Episode {episode_number} transcribed successfully")
            logger.info(f"  Duration: {whisper_result.get('duration', 0):.1f}s")
            logger.info(f"  Segments: {len(segments)}")
            logger.info(f"  Saved to: {output_file}")

            # Clean up audio file to save disk space
            if not os.getenv("KEEP_AUDIO_FILES"):
                audio_path.unlink()
                logger.info(f"  Cleaned up audio file")

            return episode_data

        except Exception as e:
            logger.error(f"Failed to transcribe episode {episode_number}: {e}")
            raise


def transcribe_all_episodes(start_episode: int = 1, end_episode: int = 172, skip_existing: bool = True):
    """Transcribe all episodes in the collection"""
    transcriber = EpisodeTranscriber()

    logger.info(f"\n{'='*70}")
    logger.info(f"TRANSCRIBING EPISODES {start_episode}-{end_episode}")
    logger.info(f"{'='*70}\n")

    success_count = 0
    error_count = 0
    skip_count = 0

    for episode_num in range(start_episode, end_episode + 1):
        # Skip episode 26 (doesn't exist)
        if episode_num == 26:
            logger.info(f"Skipping episode 26 (known gap)")
            skip_count += 1
            continue

        try:
            # Load existing episode to get audio URL
            episode_file = TRANSCRIPTS_DIR / f"episode_{episode_num}.json"

            if not episode_file.exists():
                logger.warning(f"Episode {episode_num} file not found, skipping")
                skip_count += 1
                continue

            with open(episode_file) as f:
                episode_data = json.load(f)

            audio_url = episode_data.get('audio_url')
            if not audio_url:
                logger.warning(f"Episode {episode_num} has no audio URL, skipping")
                skip_count += 1
                continue

            # Transcribe episode
            transcriber.transcribe_episode(episode_num, audio_url, skip_if_exists=skip_existing)
            success_count += 1

        except Exception as e:
            logger.error(f"Error processing episode {episode_num}: {e}")
            error_count += 1
            continue

    logger.info(f"\n{'='*70}")
    logger.info(f"TRANSCRIPTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Skipped: {skip_count}")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Test on first episode only
            logger.info("Running in TEST mode (episode 1 only)")
            transcribe_all_episodes(start_episode=1, end_episode=1, skip_existing=False)
        elif sys.argv[1] == "range":
            # Transcribe a range
            start = int(sys.argv[2])
            end = int(sys.argv[3])
            logger.info(f"Transcribing episodes {start}-{end}")
            transcribe_all_episodes(start_episode=start, end_episode=end, skip_existing=True)
        else:
            logger.error("Usage: python retranscribe_episodes_lightweight.py [test|range START END]")
            sys.exit(1)
    else:
        # Transcribe all episodes
        transcribe_all_episodes(start_episode=1, end_episode=172, skip_existing=True)
