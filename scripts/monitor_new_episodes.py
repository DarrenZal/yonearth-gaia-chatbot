#!/usr/bin/env python3
"""
Monitor YonEarth podcast RSS feed for new episodes and process them.

This script:
1. Checks the RSS feed for new episodes
2. Scrapes episode pages for audio URLs
3. Transcribes with Whisper (word-level timestamps)
4. Chunks and adds to vector store
5. Extracts knowledge graph entities

Run manually or via cron:
    # Weekly check (Sundays at 2am)
    0 2 * * 0 cd /root/yonearth-gaia-chatbot && /usr/bin/python3 scripts/monitor_new_episodes.py >> /var/log/yonearth-monitor.log 2>&1
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/yonearth-monitor.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RSS_FEED_URL = "https://yonearth.org/podcast/feed/"
TRANSCRIPTS_DIR = Path("/root/yonearth-gaia-chatbot/data/transcripts")
PROCESSED_EPISODES_FILE = Path("/root/yonearth-gaia-chatbot/data/processed/processed_episodes.json")

def get_existing_episodes():
    """Get list of episode numbers we already have."""
    existing = set()
    for f in TRANSCRIPTS_DIR.glob("episode_*.json"):
        match = re.search(r'episode_(\d+)', f.name)
        if match:
            existing.add(int(match.group(1)))
    return existing

def parse_rss_feed():
    """Parse RSS feed and return episode info."""
    logger.info(f"Fetching RSS feed: {RSS_FEED_URL}")
    feed = feedparser.parse(RSS_FEED_URL)

    episodes = []
    for entry in feed.entries:
        # Extract episode number from title
        match = re.search(r'Episode\s+(\d+)', entry.title, re.IGNORECASE)
        if match:
            ep_num = int(match.group(1))
            episodes.append({
                'number': ep_num,
                'title': entry.title,
                'url': entry.link,
                'published': entry.get('published', ''),
                'description': entry.get('summary', '')[:500]
            })

    return episodes

def scrape_episode_page(episode_url, episode_title):
    """Scrape episode page for full metadata matching existing episode format."""
    logger.info(f"Scraping episode page: {episode_url}")

    headers = {'User-Agent': 'Mozilla/5.0 (compatible; YonEarthBot/1.0)'}
    response = requests.get(episode_url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract title from page (may be cleaner than RSS)
    title_elem = soup.find('h1', class_='entry-title') or soup.find('h1')
    page_title = title_elem.get_text(strip=True) if title_elem else episode_title

    # Extract guest name from title (pattern: "Episode N – Guest Name, Title/Role")
    guest_name = ""
    title_match = re.search(r'Episode\s+\d+\s*[-–—]\s*(.+?)(?:,|$)', page_title)
    if title_match:
        guest_name = title_match.group(1).strip()

    # Extract publish date
    date_elem = soup.find('time') or soup.find(class_='entry-date')
    publish_date = ""
    if date_elem:
        publish_date = date_elem.get('datetime', date_elem.get_text(strip=True))

    # Find audio URL
    audio_url = ""
    audio_elem = soup.find('audio')
    if audio_elem:
        source = audio_elem.find('source')
        if source:
            audio_url = source.get('src', '')

    if not audio_url:
        for link in soup.find_all('a', href=True):
            if '.mp3' in link['href']:
                audio_url = link['href']
                break

    # Extract content sections
    content_elem = soup.find(class_='entry-content') or soup.find('article')
    subtitle = ""
    description = ""
    full_text = ""

    if content_elem:
        paragraphs = content_elem.find_all('p')
        if paragraphs:
            subtitle = paragraphs[0].get_text(strip=True)[:200]
        description = content_elem.get_text(strip=True)[:1000]
        full_text = content_elem.get_text(strip=True)

    # Extract "About" sections (guest bios, org info)
    about_sections = {}
    for heading in soup.find_all(['h2', 'h3']):
        heading_text = heading.get_text(strip=True)
        if 'about' in heading_text.lower():
            content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h2', 'h3']:
                    break
                if sibling.name == 'p':
                    content.append(sibling.get_text(strip=True))
            if content:
                key = heading_text.lower().replace(' ', '_').replace('.', '')
                about_sections[key] = ' '.join(content)

    # Extract related episodes
    related_episodes = []
    related_section = soup.find(string=re.compile(r'related episode', re.I))
    if related_section:
        parent = related_section.find_parent()
        if parent:
            for link in parent.find_all('a', href=True):
                related_episodes.append({
                    'title': link.get_text(strip=True),
                    'url': link['href']
                })

    # Extract sponsors
    sponsors = ""
    sponsors_section = soup.find(string=re.compile(r'sponsor', re.I))
    if sponsors_section:
        parent = sponsors_section.find_parent()
        if parent:
            sponsors = parent.get_text(strip=True)

    # Extract categories/tags from page
    categories = []
    for cat_link in soup.find_all('a', rel='category tag'):
        categories.append(cat_link.get_text(strip=True))

    return {
        'title': page_title,
        'guest_name': guest_name,
        'audio_url': audio_url,
        'publish_date': publish_date,
        'subtitle': subtitle,
        'description': description,
        'about_sections': about_sections,
        'sponsors': sponsors,
        'related_episodes': related_episodes,
        'categories': categories,
        'page_content': full_text
    }

def transcribe_episode(audio_url, episode_num):
    """Transcribe audio using Whisper with word-level timestamps."""
    logger.info(f"Transcribing episode {episode_num} from {audio_url}")

    # Download audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        response = requests.get(audio_url, stream=True, timeout=300)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Use Whisper for transcription
        import whisper

        model = whisper.load_model("base")  # Use base for speed, medium for quality
        result = model.transcribe(
            tmp_path,
            word_timestamps=True,
            language="en"
        )

        return result
    finally:
        os.unlink(tmp_path)

def save_transcript(episode_info, transcript_result, page_data):
    """Save transcript with full metadata matching existing episode format."""
    episode_num = episode_info['number']

    # Build segments with timestamps
    segments = []
    if transcript_result and 'segments' in transcript_result:
        for seg in transcript_result['segments']:
            segments.append({
                'id': seg.get('id', 0),
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'words': seg.get('words', [])
            })

    # Use page-scraped title if available (often cleaner)
    title = page_data.get('title', episode_info['title'])

    transcript_data = {
        # Core fields (matching existing format)
        'title': title,
        'audio_url': page_data.get('audio_url', ''),
        'publish_date': page_data.get('publish_date', episode_info['published']),
        'url': episode_info['url'],
        'episode_number': episode_num,
        'subtitle': page_data.get('subtitle', ''),
        'description': page_data.get('description', episode_info['description']),
        'guest_name': page_data.get('guest_name', ''),
        'about_sections': page_data.get('about_sections', {}),
        'sponsors': page_data.get('sponsors', ''),
        'related_episodes': page_data.get('related_episodes', []),
        'categories': page_data.get('categories', []),

        # Transcript content
        'full_transcript': transcript_result.get('text', page_data['page_content']) if transcript_result else page_data['page_content'],
        'segments': segments,

        # Transcription metadata
        'audio_transcription_metadata': {
            'whisper_model': 'base' if transcript_result else None,
            'language': transcript_result.get('language', 'en') if transcript_result else 'en',
            'segments_count': len(segments),
            'transcription_method': 'whisper-base' if transcript_result else 'page-scrape',
            'transcribed_at': datetime.now().isoformat()
        }
    }

    output_path = TRANSCRIPTS_DIR / f"episode_{episode_num}.json"
    with open(output_path, 'w') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved transcript to {output_path}")
    logger.info(f"  - Guest: {transcript_data['guest_name']}")
    logger.info(f"  - Audio URL: {'Yes' if transcript_data['audio_url'] else 'No'}")
    logger.info(f"  - Segments: {len(segments)}")
    return output_path

def process_episode_pipeline(episode_num):
    """Run the full processing pipeline for a new episode."""
    logger.info(f"Running processing pipeline for episode {episode_num}")

    # 1. Process episode (chunk + vectorize)
    try:
        subprocess.run([
            sys.executable, "-m", "src.ingestion.process_episodes",
        ], env={**os.environ, 'EPISODES_TO_PROCESS': str(episode_num)}, check=True, cwd="/root/yonearth-gaia-chatbot")
        logger.info(f"Episode {episode_num} added to vector store")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process episode {episode_num}: {e}")

    # 2. Extract knowledge graph (optional - can be resource intensive)
    try:
        subprocess.run([
            sys.executable, "scripts/extract_kg_v3_2_2.py",
            "--episodes", str(episode_num)
        ], check=True, cwd="/root/yonearth-gaia-chatbot")
        logger.info(f"Knowledge graph extracted for episode {episode_num}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"KG extraction failed for episode {episode_num}: {e}")

def update_processed_log(episode_num, status):
    """Track which episodes have been processed."""
    if PROCESSED_EPISODES_FILE.exists():
        with open(PROCESSED_EPISODES_FILE) as f:
            processed = json.load(f)
    else:
        processed = {'episodes': {}}

    processed['episodes'][str(episode_num)] = {
        'status': status,
        'processed_at': datetime.now().isoformat()
    }

    with open(PROCESSED_EPISODES_FILE, 'w') as f:
        json.dump(processed, f, indent=2)

def main():
    """Main monitoring loop."""
    logger.info("=" * 60)
    logger.info("YonEarth Podcast Monitor - Starting check")
    logger.info("=" * 60)

    # Get existing episodes
    existing = get_existing_episodes()
    logger.info(f"Found {len(existing)} existing episodes (latest: {max(existing) if existing else 'none'})")

    # Parse RSS feed
    feed_episodes = parse_rss_feed()
    logger.info(f"Found {len(feed_episodes)} episodes in RSS feed")

    # Find new episodes
    new_episodes = [ep for ep in feed_episodes if ep['number'] not in existing]

    if not new_episodes:
        logger.info("No new episodes found")
        return

    logger.info(f"Found {len(new_episodes)} new episodes: {[ep['number'] for ep in new_episodes]}")

    # Process each new episode
    for episode in sorted(new_episodes, key=lambda x: x['number']):
        ep_num = episode['number']
        logger.info(f"\n{'='*40}\nProcessing Episode {ep_num}: {episode['title']}\n{'='*40}")

        try:
            # 1. Scrape episode page for full metadata
            page_data = scrape_episode_page(episode['url'], episode['title'])

            # 2. Transcribe if audio available
            transcript_result = None
            if page_data['audio_url']:
                try:
                    transcript_result = transcribe_episode(page_data['audio_url'], ep_num)
                except Exception as e:
                    logger.warning(f"Transcription failed, using page content: {e}")

            # 3. Save transcript
            save_transcript(episode, transcript_result, page_data)

            # 4. Run processing pipeline
            process_episode_pipeline(ep_num)

            # 5. Update log
            update_processed_log(ep_num, 'success')

            logger.info(f"Episode {ep_num} processed successfully!")

        except Exception as e:
            logger.error(f"Failed to process episode {ep_num}: {e}")
            update_processed_log(ep_num, f'failed: {str(e)}')

    logger.info("\n" + "=" * 60)
    logger.info("Monitoring complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
