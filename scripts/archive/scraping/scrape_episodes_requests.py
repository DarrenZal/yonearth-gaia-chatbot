#!/usr/bin/env python3
"""
Scrape missing episodes from yonearth.org using requests and BeautifulSoup.

This replicates the original scraping process that was used to download
the 164 episodes we currently have.
"""

import json
import logging
import time
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Episodes to scrape
MISSING_EPISODES = [16, 58, 101, 105]

# Output directory
OUTPUT_DIR = Path('/Users/darrenzal/projects/yonearth-gaia-chatbot/data/transcripts')


def build_episode_url(episode_num):
    """Build the episode URL - we need to find the slug."""
    # For now, return None - we'll search for it
    return None


def search_for_episode(episode_num):
    """Search the podcast archive for the episode URL."""
    logger.info(f"Searching for episode {episode_num} URL...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Try direct URL patterns first
    possible_patterns = [
        f"https://yonearth.org/podcast/episode-{episode_num:03d}-",
        f"https://yonearth.org/podcast/episode-{episode_num}-",
    ]

    # Search the main podcast page
    try:
        response = requests.get("https://yonearth.org/podcast/", headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find links containing episode number
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if f'episode-{episode_num:03d}' in href or f'episode-{episode_num}-' in href:
                logger.info(f"  Found URL: {href}")
                return href

    except Exception as e:
        logger.error(f"Error searching for episode {episode_num}: {e}")

    return None


def scrape_episode(episode_num):
    """Scrape a single episode."""

    # Find episode URL
    episode_url = search_for_episode(episode_num)
    if not episode_url:
        logger.error(f"Could not find URL for episode {episode_num}")
        return None

    logger.info(f"Scraping episode {episode_num} from {episode_url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        # Fetch episode page
        response = requests.get(episode_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_elem = soup.find('h1', class_='entry-title') or soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else f"Episode {episode_num}"

        # Extract publish date
        date_elem = soup.find('time') or soup.find(class_='entry-date')
        publish_date = "Unknown"
        if date_elem:
            publish_date = date_elem.get('datetime', date_elem.get_text(strip=True))

        # Extract audio URL
        audio_url = ""
        # Try audio element
        audio_elem = soup.find('audio')
        if audio_elem:
            source = audio_elem.find('source')
            if source:
                audio_url = source.get('src', '')

        # Try MP3 links
        if not audio_url:
            for link in soup.find_all('a', href=True):
                if '.mp3' in link['href']:
                    audio_url = link['href']
                    break

        # Extract content
        content_elem = soup.find(class_='entry-content') or soup.find('article')
        description = ""
        subtitle = ""
        full_transcript = ""

        if content_elem:
            # Get all paragraphs
            paragraphs = content_elem.find_all('p')

            # First paragraph as subtitle
            if paragraphs:
                subtitle = paragraphs[0].get_text(strip=True)[:200]

            # All text as description
            description = content_elem.get_text(strip=True)[:1000]

            # Try to find transcript
            transcript_section = content_elem.find(string=re.compile(r'transcript', re.I))
            if transcript_section:
                # Get text after "transcript" heading
                parent = transcript_section.parent
                full_transcript = parent.get_text(strip=True)
            else:
                # Use full content as transcript
                full_transcript = content_elem.get_text(strip=True)

        # Extract about sections
        about_sections = {}
        for heading in soup.find_all(['h2', 'h3']):
            heading_text = heading.get_text(strip=True)
            if 'about' in heading_text.lower():
                # Get next sibling paragraphs
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

        # Build episode data
        episode_data = {
            'title': title,
            'audio_url': audio_url,
            'publish_date': publish_date,
            'url': episode_url,
            'episode_number': episode_num,
            'subtitle': subtitle,
            'description': description,
            'about_sections': about_sections,
            'sponsors': sponsors,
            'related_episodes': related_episodes,
            'full_transcript': full_transcript
        }

        logger.info(f"✅ Successfully scraped episode {episode_num}")
        return episode_data

    except Exception as e:
        logger.error(f"Error scraping episode {episode_num}: {e}", exc_info=True)
        return None


def main():
    """Main scraping process."""
    logger.info("="*80)
    logger.info("SCRAPING MISSING EPISODES FROM YONEARTH.ORG")
    logger.info("="*80)
    logger.info(f"Episodes to scrape: {MISSING_EPISODES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = []

    for episode_num in MISSING_EPISODES:
        logger.info(f"\nProcessing episode {episode_num}...")

        try:
            episode_data = scrape_episode(episode_num)

            if episode_data:
                # Save to JSON file
                output_file = OUTPUT_DIR / f'episode_{episode_num}.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(episode_data, f, indent=2, ensure_ascii=False)

                logger.info(f"  💾 Saved to {output_file}")
                successful += 1
            else:
                failed.append(episode_num)

            # Rate limiting
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing episode {episode_num}: {e}", exc_info=True)
            failed.append(episode_num)

    # Summary
    logger.info("="*80)
    logger.info("SCRAPING COMPLETE")
    logger.info("="*80)
    logger.info(f"Successful: {successful}/{len(MISSING_EPISODES)}")
    if failed:
        logger.info(f"Failed: {failed}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
