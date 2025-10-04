#!/usr/bin/env python3
"""
Simple static site generator for the knowledge graph wiki.
Creates a browseable HTML version of the markdown wiki with wikilinks.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set
import markdown
from jinja2 import Template

# Directories
WIKI_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/data/knowledge_graph/wiki")
OUTPUT_DIR = Path("/home/claudeuser/yonearth-gaia-chatbot/web/wiki")

# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - YonEarth Knowledge Wiki</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 250px 1fr;
            gap: 20px;
            padding: 20px;
            min-height: 100vh;
        }

        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            max-height: calc(100vh - 40px);
            overflow-y: auto;
            position: sticky;
            top: 20px;
        }

        .sidebar h2 {
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }

        .sidebar nav a {
            display: block;
            padding: 8px 12px;
            color: #555;
            text-decoration: none;
            border-radius: 6px;
            transition: all 0.2s;
            font-size: 0.9rem;
        }

        .sidebar nav a:hover {
            background: #667eea;
            color: white;
            transform: translateX(5px);
        }

        .content {
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            min-height: calc(100vh - 40px);
        }

        .content h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }

        .content h2 {
            color: #764ba2;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.8rem;
        }

        .content h3 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .content a {
            color: #667eea;
            text-decoration: none;
            border-bottom: 1px solid #667eea;
            transition: all 0.2s;
        }

        .content a:hover {
            color: #764ba2;
            border-bottom-color: #764ba2;
        }

        .metadata {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
        }

        .metadata strong {
            color: #667eea;
        }

        .badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            margin: 4px;
        }

        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            padding: 8px 16px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: all 0.2s;
        }

        .back-link:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .search-box {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }

        .search-box:focus {
            outline: none;
            border-color: #667eea;
        }

        ul, ol {
            margin-left: 30px;
            margin-bottom: 15px;
        }

        li {
            margin-bottom: 8px;
        }

        code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 15px 0;
        }

        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }

            .sidebar {
                position: static;
                max-height: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <h2>🗺️ Navigation</h2>
            <input type="text" class="search-box" placeholder="Search wiki..." id="search">
            <nav>
                <a href="/wiki/index.html">🏠 Home</a>
                <a href="/wiki/people/index.html">👥 People</a>
                <a href="/wiki/organizations/index.html">🏢 Organizations</a>
                <a href="/wiki/concepts/index.html">💡 Concepts</a>
                <a href="/wiki/practices/index.html">🛠️ Practices</a>
                <a href="/wiki/technologies/index.html">⚙️ Technologies</a>
                <a href="/wiki/locations/index.html">📍 Locations</a>
                <a href="/wiki/episodes/index.html">🎙️ Episodes</a>
                <a href="/knowledge-graph">🕸️ Knowledge Graph</a>
            </nav>
        </aside>

        <main class="content">
            {{ content }}
        </main>
    </div>

    <script>
        // Simple client-side search
        document.getElementById('search').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const links = document.querySelectorAll('.sidebar nav a');

            links.forEach(link => {
                const text = link.textContent.toLowerCase();
                if (text.includes(query) || query === '') {
                    link.style.display = 'block';
                } else {
                    link.style.display = 'none';
                }
            });
        });
    </script>
</body>
</html>
"""


def convert_wikilinks_to_html(content: str, current_dir: str) -> str:
    """Convert [[WikiLink]] to HTML links."""
    def replace_link(match):
        link_text = match.group(1)
        # Remove any markdown formatting
        link_text = link_text.strip()

        # Sanitize for filename
        filename = link_text.replace(' ', '_').replace('/', '_')

        # Try to find the file (check in common locations)
        # For now, just link to search or use # anchor
        return f'<a href="/wiki/search.html?q={link_text}" class="wikilink">{link_text}</a>'

    # Replace [[link]] patterns
    content = re.sub(r'\[\[([^\]]+)\]\]', replace_link, content)
    return content


def process_markdown_file(md_path: Path, output_path: Path, entity_type: str = ""):
    """Convert a markdown file to HTML."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert wikilinks to HTML
        content = convert_wikilinks_to_html(content, md_path.parent.name)

        # Convert markdown to HTML
        html_content = markdown.markdown(content, extensions=['extra', 'codehilite', 'toc'])

        # Extract title from first heading or filename
        title_match = re.search(r'<h1>(.*?)</h1>', html_content)
        if title_match:
            title = title_match.group(1)
        else:
            title = md_path.stem.replace('_', ' ')

        # Render template
        template = Template(HTML_TEMPLATE)
        full_html = template.render(title=title, content=html_content)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)

        return True
    except Exception as e:
        print(f"Error processing {md_path}: {e}")
        return False


def build_site():
    """Build the complete static site."""
    print("🏗️  Building YonEarth Knowledge Wiki...")

    # Clean output directory
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all markdown files
    md_files = list(WIKI_DIR.rglob("*.md"))
    print(f"📄 Found {len(md_files)} markdown files")

    # Process each file
    successful = 0
    for md_file in md_files:
        # Get relative path
        rel_path = md_file.relative_to(WIKI_DIR)

        # Create output path (change .md to .html)
        html_path = OUTPUT_DIR / rel_path.with_suffix('.html')

        # Get entity type from directory
        entity_type = rel_path.parts[0] if len(rel_path.parts) > 1 else ""

        if process_markdown_file(md_file, html_path, entity_type):
            successful += 1

    print(f"✅ Successfully processed {successful}/{len(md_files)} files")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🌐 Access at: http://152.53.194.214/wiki/index.html")


if __name__ == "__main__":
    build_site()
