# Our Biggest Deal - Knowledge Graph Visualization

## Overview

A custom knowledge graph visualization specifically for Aaron William Perry's book "Our Biggest Deal", built using the same D3.js force-directed graph technology as the YonEarth podcast knowledge graph.

## Current Status

**✅ COMPLETE - Front Matter Extraction (Pages 1-30)**

- **167 entities** extracted from the front matter
- **300 relationships** mapped between entities
- **6 domains**: People, Organizations, Publications, Concepts, Events, Places
- **30 entity types**: Person, Book, Organization, Essay, Event, Concept, and more

## Features

### Interactive Visualization
- Force-directed graph layout with D3.js
- Pan, zoom, and drag interactions
- Node sizing based on entity importance
- Color coding by domain category
- Click nodes to view detailed information

### Filtering & Search
- **Domain filtering**: Show/hide specific categories
- **Entity type filtering**: Filter by Person, Book, Organization, etc.
- **Importance threshold**: Adjust minimum importance level (default: 0.5)
- **Text search**: Find entities by name or description

### Entity Details Panel
- View entity metadata (type, domain, description)
- See all relationships (incoming and outgoing)
- Page references for each relationship
- Mention count and page count

### Layout Controls
- **Gravity**: Adjust center force (0.0-0.5)
- **Charge**: Control node repulsion (-1000 to -50)
- **Link Distance**: Adjust connection length (20-300)
- **Reset View**: Return to initial layout

## Files Created

### Web Interface
- **`/web/OurBiggestDealKG.html`** - Main HTML page (9.1KB)
- **`/web/OurBiggestDealKG.js`** - Visualization JavaScript (27KB)
- Reuses existing **`/web/KnowledgeGraph.css`** for styling

### Data Files
- **`/data/knowledge_graph_books/our_biggest_deal_visualization.json`** - Transformed visualization data (152KB)

### Scripts
- **`/scripts/transform_book_kg_for_viz.py`** - Transforms extracted KG data into D3.js format

## Data Pipeline

### 1. Extraction (Already Complete)
```bash
# Extract KG from book chapters
python3 scripts/extract_kg_v14_3_8_incremental.py \
  --book our_biggest_deal \
  --section front_matter \
  --pages 1-30 \
  --author "Aaron William Perry"
```

**Output**: `/kg_extraction_playbook/output/our_biggest_deal/v14_3_8/chapters/front_matter_v14_3_8_*.json`

### 2. Transformation (Run Anytime)
```bash
# Transform extracted data for visualization
python3 scripts/transform_book_kg_for_viz.py
```

**Output**: `/data/knowledge_graph_books/our_biggest_deal_visualization.json`

### 3. Deployment
```bash
# Development: Files already in /web directory, accessible via local server

# Production: Copy to production server
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/OurBiggestDealKG.* /var/www/yonearth/
sudo cp -r /home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_books /var/www/yonearth/data/
sudo systemctl reload nginx
```

## Accessing the Visualization

### Local Development
```
http://localhost:8000/OurBiggestDealKG.html
```

### Production
```
http://152.53.194.214/OurBiggestDealKG.html
```

## Current Data Statistics

**Front Matter Section (Pages 1-30)**

| Metric | Value |
|--------|-------|
| Total Entities | 167 |
| Total Relationships | 300 |
| Entity Types | 30 |
| Domains | 6 |
| Pages Covered | 29 |
| Extraction Version | v14_3_8 |

### Domain Breakdown
- **People** - 77 persons (Yvon Chouinard, Ken LaRoe, Samantha Power, etc.)
- **Publications** - 34 books, 1 essay, 2 quotes
- **Concepts** - 15 concepts, ideas, frameworks
- **Organizations** - 5 organizations, 1 foundation, 1 bank
- **Events** - 6 events
- **Places** - 2 locations

### Top Relationship Types
- `endorsed` - Endorsements of the book
- `wrote` - Authorship relationships
- `inspired` - Inspirational connections
- `founded` - Organizational founding
- `published` - Publication relationships

## Expanding the Visualization

### Adding More Chapters

1. **Extract the chapter**:
```bash
python3 scripts/extract_kg_v14_3_8_incremental.py \
  --book our_biggest_deal \
  --section chapter_1 \
  --pages 31-60 \
  --author "Aaron William Perry"
```

2. **Regenerate visualization data**:
```bash
python3 scripts/transform_book_kg_for_viz.py
```

The transformation script automatically finds and combines all available chapter extractions.

3. **Update section selector** in `OurBiggestDealKG.html`:
```html
<select id="section-select" onchange="loadBookSection(this.value)">
    <option value="front_matter">Front Matter (Pages 1-30)</option>
    <option value="chapter_1">Chapter 1 (Pages 31-60)</option>
    <!-- Add more chapters as extracted -->
</select>
```

## Comparison with YonEarth KG

| Feature | YonEarth Podcast KG | Our Biggest Deal Book KG |
|---------|---------------------|--------------------------|
| Data Source | 172 podcast episodes | Book chapters |
| Total Entities | 10,000+ | 167 (front matter only) |
| Reference Type | Episode numbers | Page numbers |
| Domains | 8+ podcast topics | 6 book domains |
| Importance Threshold | 0.7 (high) | 0.5 (lower for smaller dataset) |
| Max Nodes | 1,000 | 500 |

## Technical Details

### Data Format (Input)
The extraction produces JSON files with this structure:
```json
{
  "metadata": {
    "book": "our_biggest_deal",
    "section": "front_matter",
    "pages": "1-30",
    "extraction_version": "v14_3_8"
  },
  "relationships": [
    {
      "source": "Ken LaRoe",
      "relationship": "endorsed",
      "target": "Our Biggest Deal",
      "source_type": "Person",
      "target_type": "Book",
      "context": "...",
      "page": 3,
      "p_true": 0.9
    }
  ]
}
```

### Data Format (Visualization)
The transformation script converts to D3.js format:
```json
{
  "nodes": [
    {
      "id": "Ken LaRoe",
      "name": "Ken LaRoe",
      "type": "Person",
      "domains": ["People"],
      "domain_colors": ["#44e5e5"],
      "importance": 0.85,
      "mention_count": 3,
      "page_count": 2,
      "pages": [3, 5],
      "description": "..."
    }
  ],
  "links": [
    {
      "source": "Ken LaRoe",
      "target": "Our Biggest Deal",
      "type": "endorsed",
      "strength": 0.9,
      "page": 3
    }
  ],
  "domains": [...],
  "entity_types": [...],
  "statistics": {...}
}
```

## Future Enhancements

### Planned Features
- [ ] Load multiple book sections dynamically
- [ ] Timeline view showing entity mentions across pages
- [ ] Export functionality (PNG, SVG, JSON)
- [ ] Entity clustering by similarity
- [ ] Integration with chat interface for contextual queries
- [ ] Cross-reference with YonEarth podcast entities

### Data Expansion
- [ ] Extract remaining chapters sequentially
- [ ] Build complete book knowledge graph
- [ ] Add author annotations and insights
- [ ] Link to external resources (Wikipedia, book pages)

## Troubleshooting

### Visualization Not Loading
1. Check browser console for errors
2. Verify data file exists: `/data/knowledge_graph_books/our_biggest_deal_visualization.json`
3. Ensure local server is running or files are deployed

### Empty or Broken Graph
1. Verify extraction completed successfully
2. Check transformation script output for errors
3. Confirm JSON data is valid (use `jq` or JSON validator)

### Performance Issues
1. Increase importance threshold to reduce node count
2. Disable less relevant domains/entity types
3. Adjust max nodes limit in filters

## Support & Maintenance

### Regenerating Visualization Data
Run anytime after new extractions:
```bash
python3 scripts/transform_book_kg_for_viz.py
```

### Updating Production
```bash
# Update web files
sudo cp /home/claudeuser/yonearth-gaia-chatbot/web/OurBiggestDealKG.* /var/www/yonearth/

# Update data files
sudo cp -r /home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph_books /var/www/yonearth/data/

# Reload nginx (cache busting)
sudo systemctl reload nginx
```

### Monitoring Extraction Progress
Check the extraction logs:
```bash
tail -f kg_extraction_v14_3_8_front_matter.log
```

## Credits

- **Book**: "Our Biggest Deal" by Aaron William Perry
- **Visualization**: Based on YonEarth podcast KG by the same team
- **Technology**: D3.js force-directed graph, OpenAI GPT-4 extraction
- **Extraction Framework**: KGC (Knowledge Graph Construction) v14.3.8

---

**Last Updated**: October 15, 2025
**Status**: Front Matter Complete, Full Book In Progress
