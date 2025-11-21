"""
Episode processor for loading and preparing podcast episodes
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)


class Episode:
    """Represents a podcast episode with metadata and content"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data  # Store raw data for access to segments
        self.episode_number = data.get("episode_number", "unknown")
        self.title = data.get("title", "")
        self.subtitle = data.get("subtitle", "")
        self.description = data.get("description", "")
        self.guest_name = self._extract_guest_name(data)
        self.publish_date = data.get("publish_date", "")
        self.audio_url = data.get("audio_url", "")
        self.url = data.get("url", "")
        self.transcript = data.get("full_transcript", "")
        self.about_sections = data.get("about_sections", {})
        self.related_episodes = data.get("related_episodes", [])
        
    def _extract_guest_name(self, data: Dict[str, Any]) -> str:
        """Extract guest name from various possible locations"""
        # Try to extract from about_guest section
        about_guest = data.get("about_sections", {}).get("about_guest", "")
        if about_guest:
            # Simple heuristic: first sentence often contains guest name
            first_sentence = about_guest.split(".")[0]
            if "is" in first_sentence:
                return first_sentence.split("is")[0].strip()
        
        # Try to extract from subtitle
        subtitle = data.get("subtitle", "")
        if " with " in subtitle:
            return subtitle.split(" with ")[-1].strip()
        
        return "Guest"
    
    @property
    def has_transcript(self) -> bool:
        """Check if episode has a valid transcript"""
        return bool(self.transcript and len(self.transcript) > 100)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get episode metadata for vector storage"""
        return {
            "episode_number": str(self.episode_number),
            "title": self.title,
            "guest_name": self.guest_name,
            "publish_date": self.publish_date,
            "url": self.url,
            "audio_url": self.audio_url,
            "subtitle": self.subtitle
        }
    
    def to_document(self) -> Dict[str, Any]:
        """Convert episode to document format for vector storage"""
        return {
            "page_content": self.transcript,
            "metadata": self.metadata
        }


class EpisodeProcessor:
    """Process podcast episodes for ingestion into vector database"""
    
    def __init__(self):
        self.episodes_dir = settings.episodes_dir
        self.processed_dir = settings.processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_episodes(self, limit: Optional[int] = None) -> List[Episode]:
        """Load episodes from JSON files"""
        episode_files = settings.get_episode_files(limit=limit or settings.episodes_to_process)
        episodes = []
        
        for file_path in episode_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                episode = Episode(data)
                
                if episode.has_transcript:
                    episodes.append(episode)
                    logger.info(f"Loaded episode {episode.episode_number}: {episode.title}")
                else:
                    logger.warning(f"Skipping episode {episode.episode_number} - no transcript")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        logger.info(f"Loaded {len(episodes)} episodes with transcripts")
        return episodes
    
    def get_diverse_episodes(self, episodes: List[Episode], count: int = 20) -> List[Episode]:
        """Select diverse episodes for MVP (different topics, guests, time periods)"""
        if len(episodes) <= count:
            return episodes
            
        # Simple diversity selection: spread across episode numbers
        step = len(episodes) // count
        diverse_episodes = []
        
        for i in range(0, len(episodes), step):
            if len(diverse_episodes) < count:
                diverse_episodes.append(episodes[i])
                
        logger.info(f"Selected {len(diverse_episodes)} diverse episodes for processing")
        return diverse_episodes
    
    def save_processed_episodes(self, episodes: List[Episode]):
        """Save processed episodes metadata for reference"""
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "episode_count": len(episodes),
            "episodes": [
                {
                    "number": ep.episode_number,
                    "title": ep.title,
                    "guest": ep.guest_name,
                    "date": ep.publish_date,
                    "transcript_length": len(ep.transcript)
                }
                for ep in episodes
            ]
        }
        
        output_path = self.processed_dir / "episode_metadata.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved episode metadata to {output_path}")


def main():
    """Test episode processing"""
    logging.basicConfig(level=logging.INFO)
    
    processor = EpisodeProcessor()
    episodes = processor.load_episodes(limit=5)
    
    for episode in episodes:
        print(f"\nEpisode {episode.episode_number}: {episode.title}")
        print(f"Guest: {episode.guest_name}")
        print(f"Transcript length: {len(episode.transcript)} characters")
        print(f"Has transcript: {episode.has_transcript}")


if __name__ == "__main__":
    main()