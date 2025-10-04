# YonEarth Podcast Episode Coverage

## Complete Episode Inventory

**Total Episodes Available**: 172
**Episode Range**: 0-172
**Missing Episodes**: 1 (episode #26 was never published)

## Episode Breakdown

### Episodes 0-50
- **Episode 0**: Intro/Welcome episode
- **Episodes 1-25**: ✅ All downloaded (25 episodes)
- **Episode 26**: ❌ Does not exist (gap in numbering)
- **Episodes 27-47**: ✅ All downloaded (21 episodes)
- **Episode 48**: ✅ Downloaded (special URL: `/episode-matt-gray-cso/`)
- **Episodes 49-50**: ✅ Downloaded (2 episodes)

### Episodes 51-100
- **Episodes 51-52**: ✅ Downloaded (2 episodes)
- **Episode 53**: ✅ Downloaded (special URL: `/53-dj-spooky-paul-miller/`)
- **Episodes 54-61**: ✅ Downloaded (8 episodes)
- **Episode 62**: ✅ Downloaded (special URL: `/62-brian-kunkler/`)
- **Episode 63**: ✅ Downloaded (special URL: `/63-david-bronner/`)
- **Episodes 64-72**: ✅ Downloaded (9 episodes)
- **Episode 73**: ✅ Downloaded (special URL: `/73-sydney-harrison-steinberg-colorado-rooted/`)
- **Episode 74**: ✅ Downloaded (1 episode)
- **Episode 75**: ✅ Downloaded (special URL: `/dr-jandel-allen-davis/`)
- **Episodes 76-100**: ✅ All downloaded (25 episodes)

### Episodes 101-172
- **Episodes 101-170**: ✅ All downloaded (70 episodes)
- **Episode 171**: ✅ Downloaded (latest, special URL: `/episode-171-cynthia-james-stewart-regenerative-burgers-fries-at-james-ranch-grill/`)
- **Episode 172**: ✅ Downloaded (latest, `/episode-172-dr-riane-eisler-reclaiming-our-true-human-nature-evolving-from-domination-to-partnership-society/`)

## Special URL Patterns

### Standard Pattern
Most episodes follow this URL pattern:
```
https://yonearth.org/podcast/episode-{number}-{slug}/
```
Example: `https://yonearth.org/podcast/episode-1-nancy-tuchman/`

### Non-Standard URLs
Some episodes don't include the episode number in the URL:

| Episode | URL Pattern | Title |
|---------|-------------|-------|
| 48 | `/episode-matt-gray-cso/` | Matt Gray, Chief Sustainability Officer |
| 53 | `/53-dj-spooky-paul-miller/` | DJ Spooky (aka Paul Miller) |
| 62 | `/62-brian-kunkler/` | Pastor Brian Kunkler |
| 63 | `/63-david-bronner/` | David Bronner, CEO, Dr. Bronner's |
| 73 | `/73-sydney-harrison-steinberg-colorado-rooted/` | Sydney & Harrison Steinberg |
| 75 | `/dr-jandel-allen-davis/` | Dr. Jandel Allen Davis |

**Note**: These episodes require searching by title text, not URL pattern matching.

## Scraping Methodology

### Tools Used
- **requests** library for HTTP requests
- **BeautifulSoup4** for HTML parsing
- User-Agent headers to avoid 403 Forbidden errors
- Rate limiting (2 second delay between requests)

### Scripts
1. **`scripts/scrape_episodes_requests.py`**
   - Searches podcast archive for episodes by URL pattern
   - Downloads standard-format episodes

2. **`scripts/scrape_found_episodes.py`**
   - Downloads episodes with non-standard URLs
   - Uses hardcoded episode-to-URL mapping

### Search Strategy
```python
# Search by episode number in link text
for link in soup.find_all('a', href=True):
    text = link.get_text(strip=True)
    match = re.search(r'Episode\s+0*(\d+)', text, re.IGNORECASE)
    if match:
        episode_num = int(match.group(1))
        # Found episode!
```

## Data Format

Each episode transcript file contains:

```json
{
  "title": "Episode Title",
  "audio_url": "https://media.blubrry.com/...",
  "publish_date": "Date published",
  "url": "https://yonearth.org/podcast/...",
  "episode_number": 123,
  "subtitle": "Short description",
  "description": "Full description (first 1000 chars)",
  "about_sections": {
    "about_guest": "Guest biography",
    "about_sponsor": "Sponsor information"
  },
  "sponsors": "Sponsor text",
  "related_episodes": [
    {"title": "...", "url": "..."}
  ],
  "full_transcript": "Complete episode transcript"
}
```

## File Locations

- **Transcript Files**: `/data/transcripts/episode_{number}.json`
- **Entity Extractions**: `/data/knowledge_graph/entities/episode_{number}_extraction.json`
- **Scraping Scripts**: `/scripts/scrape_episodes_requests.py`, `/scripts/scrape_found_episodes.py`
- **Scraping Logs**: `/logs/scrape_episodes.log`

## Statistics

- **Total Episodes**: 172
- **Total Transcript Files**: 172
- **Average File Size**: ~28KB per transcript
- **Total Transcript Data**: ~4.8MB
- **Largest Transcript**: Episode 170 (Tina Morris - Bald Eagles)
- **Smallest Transcript**: Episode 0 (Welcome/Intro)

## Maintenance

### Adding New Episodes

When new episodes are published:

1. **Check for new episodes**:
   ```bash
   python3 scripts/check_new_episodes.py
   ```

2. **Download new episodes**:
   ```bash
   # For standard URL format
   python3 scripts/scrape_episodes_requests.py

   # If non-standard URL, update scrape_found_episodes.py
   ```

3. **Extract entities**:
   ```bash
   python3 scripts/extract_missing_episodes.py
   ```

4. **Rebuild knowledge graph**:
   ```bash
   python3 scripts/build_unified_knowledge_graph.py
   ```

### Verifying Coverage

```bash
# Count transcript files
ls -1 data/transcripts/*.json | wc -l

# Find missing episodes
for i in {0..172}; do
  if [ ! -f "data/transcripts/episode_$i.json" ]; then
    echo "Missing: $i"
  fi
done

# Check entity extraction coverage
ls -1 data/knowledge_graph/entities/*.json | wc -l
```

## Known Issues

### Episode 26
Episode 26 does not exist on the yonearth.org website. This is a gap in the numbering sequence and was never published.

### URL Pattern Changes
Some episodes use non-standard URL patterns without episode numbers. Always search by title text as a fallback when URL pattern matching fails.

### 403 Forbidden Errors
The website requires User-Agent headers. Without them, requests return 403 Forbidden.

**Solution**:
```python
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
requests.get(url, headers=headers)
```

## Historical Notes

- **Original Download**: July 2, 2025 (164 episodes, 0-170 with gaps)
- **Gap Episodes Found**: October 2, 2025 (episodes 48, 53, 62, 63, 73, 75)
- **Latest Episodes Added**: October 2, 2025 (episodes 171, 172)
- **Final Count**: 172 total episodes (0-172, excluding #26)

## References

- **Scraping Completion Report**: `/EPISODE_SCRAPING_COMPLETE.md`
- **Repository Cleanup Plan**: `/REPO_CLEANUP_PLAN.md`
- **Main Documentation**: `/README.md`
- **Claude Code Instructions**: `/CLAUDE.md`
