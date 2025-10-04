#!/usr/bin/env python3
"""
Scrape missing episodes from yonearth.org website.

Downloads episodes that are not yet in our transcript database:
- Episodes 26, 48, 53, 62, 63, 73, 75 (gaps in numbering)
- Episodes 171, 172 (newest episodes)
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/claudeuser/yonearth-gaia-chatbot/logs/scrape_missing_episodes.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Episodes to scrape
MISSING_EPISODES = [26, 48, 53, 62, 63, 73, 75, 171, 172]

# Output directory
OUTPUT_DIR = Path('/home/claudeuser/yonearth-gaia-chatbot/data/transcripts')


def scrape_episode(page, episode_num):
    """Scrape a single episode from yonearth.org."""

    # Episode URL pattern - need to find the exact URL
    # For now, we'll search for the episode on the site
    base_url = "https://yonearth.org/podcast"

    logger.info(f"Navigating to podcast archive to find episode {episode_num}")
    page.goto(base_url, wait_until="networkidle")
    time.sleep(2)

    # Search for episode link
    episode_link = None
    try:
        # Look for links containing the episode number
        links = page.locator(f'a[href*="episode-{episode_num:03d}"], a[href*="episode-{episode_num}"]')
        if links.count() > 0:
            episode_link = links.first.get_attribute('href')
            logger.info(f"Found episode {episode_num} at: {episode_link}")
        else:
            # Try alternative format
            links = page.locator(f'a:has-text("Episode {episode_num:03d}"), a:has-text("Episode {episode_num}")')
            if links.count() > 0:
                episode_link = links.first.get_attribute('href')
                logger.info(f"Found episode {episode_num} at: {episode_link}")
    except Exception as e:
        logger.warning(f"Could not find link for episode {episode_num}: {e}")
        return None

    if not episode_link:
        logger.error(f"Could not find episode {episode_num} in podcast archive")
        return None

    # Navigate to episode page
    page.goto(episode_link, wait_until="networkidle")
    time.sleep(2)

    # Extract episode data
    try:
        # Title
        title_element = page.locator('h1.entry-title, h1.post-title, h1')
        title = title_element.first.inner_text() if title_element.count() > 0 else f"Episode {episode_num}"

        # Publish date
        date_element = page.locator('time, .entry-date, .published')
        publish_date = date_element.first.get_attribute('datetime') if date_element.count() > 0 else "Unknown"
        if not publish_date or publish_date == "Unknown":
            publish_date = date_element.first.inner_text() if date_element.count() > 0 else "Unknown"

        # Audio URL (from blubrry player or audio element)
        audio_url = None
        audio_element = page.locator('audio source, .blubrry-player-container')
        if audio_element.count() > 0:
            audio_url = audio_element.first.get_attribute('src')

        # If no audio element, look for direct MP3 links
        if not audio_url:
            mp3_link = page.locator('a[href*=".mp3"]')
            if mp3_link.count() > 0:
                audio_url = mp3_link.first.get_attribute('href')

        # Episode content/description
        content_element = page.locator('.entry-content, .post-content, article')
        description = content_element.first.inner_text() if content_element.count() > 0 else ""

        # Extract subtitle (usually first paragraph or intro text)
        subtitle = ""
        first_p = page.locator('.entry-content p, .post-content p').first
        if first_p.count() > 0:
            subtitle = first_p.inner_text()[:200]  # First 200 chars

        # About sections (look for headings)
        about_sections = {}
        headings = page.locator('.entry-content h2, .entry-content h3, .post-content h2, .post-content h3')
        for i in range(min(headings.count(), 5)):  # Max 5 sections
            heading = headings.nth(i)
            section_title = heading.inner_text()
            if "about" in section_title.lower() or "sponsor" in section_title.lower():
                # Get content until next heading
                about_sections[section_title] = ""  # Simplified for now

        # Full transcript (if available)
        transcript = ""
        transcript_element = page.locator('.transcript, #transcript, [id*="transcript"]')
        if transcript_element.count() > 0:
            transcript = transcript_element.first.inner_text()

        # If no transcript found, use description as placeholder
        if not transcript or len(transcript) < 100:
            transcript = f"(Transcript not available on website. Content: {description[:500]})"

        episode_data = {
            'title': title.strip(),
            'audio_url': audio_url or "",
            'publish_date': publish_date,
            'url': episode_link,
            'episode_number': episode_num,
            'subtitle': subtitle.strip(),
            'description': description.strip()[:1000],  # First 1000 chars
            'about_sections': about_sections,
            'sponsors': [],
            'related_episodes': [],
            'full_transcript': transcript.strip()
        }

        logger.info(f"✅ Successfully scraped episode {episode_num}")
        return episode_data

    except Exception as e:
        logger.error(f"Error extracting data from episode {episode_num}: {e}", exc_info=True)
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

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for episode_num in MISSING_EPISODES:
            logger.info(f"\nProcessing episode {episode_num}...")

            try:
                episode_data = scrape_episode(page, episode_num)

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
                time.sleep(3)

            except Exception as e:
                logger.error(f"Error processing episode {episode_num}: {e}", exc_info=True)
                failed.append(episode_num)

        browser.close()

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
