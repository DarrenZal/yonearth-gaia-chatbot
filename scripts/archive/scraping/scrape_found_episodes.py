#!/usr/bin/env python3
"""
Scrape the newly found episodes that don't have numbers in URLs.
"""

import json
import logging
import time
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/claudeuser/yonearth-gaia-chatbot/logs/scrape_episodes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Episodes and their URLs
EPISODES = {
    48: 'https://yonearth.org/podcast/episode-matt-gray-cso/',
    53: 'https://yonearth.org/podcast/53-dj-spooky-paul-miller/',
    62: 'https://yonearth.org/podcast/62-brian-kunkler/',
    63: 'https://yonearth.org/podcast/63-david-bronner/',
    73: 'https://yonearth.org/podcast/73-sydney-harrison-steinberg-colorado-rooted/',
    75: 'https://yonearth.org/podcast/dr-jandel-allen-davis/',
}

OUTPUT_DIR = Path('/home/claudeuser/yonearth-gaia-chatbot/data/transcripts')


def scrape_episode(episode_num, episode_url):
    """Scrape a single episode."""
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

        # Extract content
        content_elem = soup.find(class_='entry-content') or soup.find('article')
        description = ""
        subtitle = ""
        full_transcript = ""

        if content_elem:
            paragraphs = content_elem.find_all('p')
            if paragraphs:
                subtitle = paragraphs[0].get_text(strip=True)[:200]

            description = content_elem.get_text(strip=True)[:1000]
            full_transcript = content_elem.get_text(strip=True)

        # Extract about sections
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
    logger.info("SCRAPING FOUND EPISODES")
    logger.info("="*80)
    logger.info(f"Episodes to scrape: {list(EPISODES.keys())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = []

    for episode_num, episode_url in EPISODES.items():
        logger.info(f"\nProcessing episode {episode_num}...")

        try:
            episode_data = scrape_episode(episode_num, episode_url)

            if episode_data:
                output_file = OUTPUT_DIR / f'episode_{episode_num}.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(episode_data, f, indent=2, ensure_ascii=False)

                logger.info(f"  💾 Saved to {output_file}")
                successful += 1
            else:
                failed.append(episode_num)

            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing episode {episode_num}: {e}", exc_info=True)
            failed.append(episode_num)

    # Summary
    logger.info("="*80)
    logger.info("SCRAPING COMPLETE")
    logger.info("="*80)
    logger.info(f"Successful: {successful}/{len(EPISODES)}")
    if failed:
        logger.info(f"Failed: {failed}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
